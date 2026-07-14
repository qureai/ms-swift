# Copyright (c) Alibaba, Inc. and its affiliates.
"""Resumable batched VLM inference with ms-swift's in-process SglangEngine.

Writes *separate* (non-overwriting) parquet files as it goes
(``<prefix>_<slug>_<timestamp>.parquet``) into an output directory, flushing
once every ``--save-every-n-batches`` inference batches, and resumes by scanning
that directory for already-processed image paths and skipping them. Modeled on
``run_inference_async_resumable.py``.

Example:
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_sglang_inference_resumable.py \
        --model /path/to/checkpoint-8241 \
        --dataset /raid4/cxr_data/testing/manifest.parquet \
        --prompt "Generate a detailed ontology report ..." \
        --output-dir /cache/.../vlm_results/my_run \
        --tp-size 8 --batch-size 5000 --save-every-n-batches 10
"""
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Set

import pandas as pd
from tqdm import tqdm

# ms-swift honors this to cap the vision token budget per image (match training).
os.environ.setdefault('MAX_PIXELS', str(2500 * 2500))

DATASET_DIR = '/mnt/nvme/legacy/raid6/cxr_data/testing/images'

WORD_BANK = [
    'alpha', 'bravo', 'charlie', 'delta', 'echo', 'foxtrot', 'golf', 'hotel', 'india', 'juliet',
    'kilo', 'lima', 'mike', 'november', 'oscar', 'papa', 'quebec', 'romeo', 'sierra', 'tango',
    'uniform', 'victor', 'whiskey', 'xray', 'yankee', 'zulu', 'ember', 'fable', 'harbor', 'island',
    'jungle', 'keeper', 'lantern', 'meadow', 'nebula', 'oracle', 'prairie', 'quartz', 'ranger',
    'saturn', 'thunder', 'utopia', 'voyager', 'willow', 'zephyr',
]


def random_word_slug(num_words: int = 4) -> str:
    """Hyphenated random words so concurrent/rerun batches never clobber each other."""
    return '-'.join(secrets.choice(WORD_BANK) for _ in range(num_words))


def resolve_image_path(x, dataset_dir):
    # Full path case
    if os.path.isabs(x):
        if x.lower().endswith(('.png', '.jpg', '.jpeg')):
            return x
        return x + '.png'
    # Relative / basename case
    fname = os.path.basename(x)  # remove any dirs if present
    if not fname.lower().endswith('.png'):
        fname += '.png'
    return os.path.join(dataset_dir, fname)


def get_message(mm_type: Literal['text', 'image'], image_path: str, prompt: str):
    if mm_type == 'text':
        return {'role': 'user', 'content': prompt}
    # image: content array (image + text). Do NOT embed a literal <image> tag here;
    # the chat template inserts the vision tokens itself.
    return {
        'role': 'user',
        'content': [
            {'type': 'image', 'image': image_path},  # url / local_path / PIL.Image / base64
            {'type': 'text', 'text': prompt},
        ],
    }


def infer_batch(engine, infer_requests, request_config, metric):
    # use_tqdm=True -> swift shows a live per-request bar (updates as each request
    # in the batch completes, with req/s == images/s). Essential for large batches,
    # otherwise the batch runs invisibly for many minutes.
    resp_list = engine.infer(infer_requests, request_config, metrics=[metric], use_tqdm=True)
    responses = [resp.choices[0].message.content for resp in resp_list]
    return responses


def load_processed_image_paths(output_dir: Path, prefix: str) -> Set[str]:
    """Scan prior batch parquets so a resumed run skips already-done images."""
    if not output_dir.exists():
        return set()
    parquet_files = sorted(output_dir.glob(f'{prefix}_*.parquet'))
    if not parquet_files:
        return set()
    processed: Set[str] = set()
    total_rows = 0
    for fp in parquet_files:
        try:
            batch_df = pd.read_parquet(fp, columns=['image_path'])
        except Exception as e:
            print(f'Warning: could not read {fp} for resume: {e}')
            continue
        total_rows += len(batch_df)
        processed.update(batch_df['image_path'].tolist())
    print(f'Resume: {total_rows} rows across {len(parquet_files)} files, '
          f'{len(processed)} unique images already done.')
    return processed


def save_buffer(buffer: List[pd.DataFrame], output_dir: Path, prefix: str) -> None:
    """Persist accumulated batches to a single new parquet (never overwrites)."""
    if not buffer:
        return
    out_df = pd.concat(buffer, ignore_index=True)
    slug = random_word_slug()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path = output_dir / f'{prefix}_{slug}_{timestamp}.parquet'
    out_df.to_parquet(path, engine='pyarrow', compression='snappy', index=False)
    print(f'Persisted {len(out_df)} rows -> {path}')


def resolve_column(df: pd.DataFrame) -> pd.DataFrame:
    if 'image_path' in df.columns:
        return df
    if 'filename' in df.columns:
        df['image_path'] = df['filename'].apply(lambda x: resolve_image_path(x, DATASET_DIR))
    elif 'image' in df.columns:
        df['image_path'] = df['image'].apply(lambda x: resolve_image_path(x, DATASET_DIR))
    else:
        raise ValueError('No image path column found in dataset (need image_path / filename / image)')
    return df


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Resumable batched sglang VLM inference (ms-swift).')
    parser.add_argument('--model', type=str, required=True, help='Model name or path')
    parser.add_argument('--dataset', type=str, required=True, help='Input CSV/Parquet dataset')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt/question for the model')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Base directory; a task_{TASK}_temperature_{TEMP}_dataset_{NAME} subdir is created under it')
    parser.add_argument('--task', type=str, required=True, help='Task name, e.g. bbox / asp')
    parser.add_argument('--dataset-name', type=str, required=True, help='Short dataset identifier, e.g. mimic_test')
    parser.add_argument('--prefix', type=str, default='vlm_results', help='Filename prefix for batch parquets')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Requests per infer() call. Keep modest (<=512): sglang concurrency is '
                             'set by --max-running-requests, and a huge batch (2000+) starves the '
                             'sglang scheduler and trips its watchdog. Use --save-every-n-batches for file size.')
    parser.add_argument('--save-every-n-batches', type=int, default=10,
                        help='Flush a parquet once every N inference batches')
    parser.add_argument('--temperature', type=float, default=0.4)
    parser.add_argument('--max-tokens', type=int, default=4096)
    # sglang engine knobs
    parser.add_argument('--tp-size', type=int, default=8, help='Tensor-parallel size')
    parser.add_argument('--dp-size', type=int, default=1, help='Data-parallel size')
    parser.add_argument('--context-length', type=int, default=8192, help='sglang max context length')
    parser.add_argument('--mem-fraction-static', type=float, default=0.8, help='sglang static mem fraction')
    parser.add_argument('--max-running-requests', type=int, default=32,
                        help='Cap concurrent decode batch (>32 can silently corrupt output on some sglang builds)')
    parser.add_argument('--limit', type=int, default=0, help='Process at most N rows (0 = all)')
    parser.add_argument('--symlink-ckpt-dir', type=str,
                        default='/cache/fast_data_nas71/janhavi/vlm_ckpts/sglang_symlink_ckpts',
                        help='Base dir for the config-fixed (symlinked) serve-ready checkpoints')
    args = parser.parse_args()

    run_subdir = f'task_{args.task}_temperature_{args.temperature}_dataset_{args.dataset_name}'
    output_dir = Path(args.output_dir).resolve() / run_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output dir: {output_dir}')

    # --- Load & prepare dataset -------------------------------------------------
    df = pd.read_csv(args.dataset) if args.dataset.endswith('.csv') else pd.read_parquet(args.dataset)
    df = resolve_column(df)
    df = df.drop_duplicates(subset=['image_path']).reset_index(drop=True)
    print(f'Dataset {args.dataset} loaded: {len(df)} unique rows')

    processed = load_processed_image_paths(output_dir, args.prefix)
    if processed:
        before = len(df)
        df = df[~df['image_path'].isin(processed)].reset_index(drop=True)
        print(f'Skipping {before - len(df)} already-processed rows; {len(df)} remain.')

    if args.limit > 0:
        df = df.head(args.limit).reset_index(drop=True)
        print(f'Limit enabled: processing first {len(df)} rows.')

    if df.empty:
        print('Nothing to do. Exiting.')
        raise SystemExit(0)

    # --- Prepare a serve-ready checkpoint --------------------------------------
    # ms-swift Qwen-VL checkpoints carry a nested text_config.model_type of
    # 'qwen2_5_vl_text', which sglang's mrope code rejects. prepare_fixed_checkpoint
    # builds a symlinked copy with a corrected config.json (original untouched).
    from prepare_qwen_checkpoint import prepare_fixed_checkpoint

    src_ckpt = Path(args.model).resolve()
    # Nest under <symlink_ckpt_dir>/<model_name>/<version>_<checkpoint> for a
    # unique, traceable path (e.g. SAHIL_..._prod/v5-20251201-063754_checkpoint-14422).
    model_name = src_ckpt.parent.parent.name or src_ckpt.name
    leaf = f'{src_ckpt.parent.name}_{src_ckpt.name}'
    dest_ckpt = Path(args.symlink_ckpt_dir).resolve() / model_name / leaf
    serve_model = prepare_fixed_checkpoint(str(src_ckpt), dest_dir=str(dest_ckpt))
    if serve_model != str(src_ckpt):
        print(f'Using config-fixed checkpoint: {serve_model}')
    else:
        print(f'Checkpoint needs no fix; serving original: {serve_model}')

    # --- Build engine -----------------------------------------------------------
    from swift import InferRequest, InferStats, RequestConfig, SglangEngine

    engine = SglangEngine(
        serve_model,
        tp_size=args.tp_size,
        dp_size=args.dp_size,
        context_length=args.context_length,
        mem_fraction_static=args.mem_fraction_static,
        engine_kwargs={'max_running_requests': args.max_running_requests},
    )
    request_config = RequestConfig(max_tokens=args.max_tokens, temperature=args.temperature)
    metric = InferStats()
    print(f'Engine ready. MAX_PIXELS={os.environ["MAX_PIXELS"]}, '
          f'temperature={args.temperature}, max_tokens={args.max_tokens}')

    # --- Inference loop: flush one parquet every N batches -----------------------
    # swift's engine.infer() draws the live per-request bar for the CURRENT batch
    # (progress + img/s); we print an overall header + throughput summary per batch.
    buffer: List[pd.DataFrame] = []
    total = len(df)
    n_batches = (total + args.batch_size - 1) // args.batch_size
    done = 0
    for batch_idx, i in enumerate(range(0, total, args.batch_size)):
        batch_df = df.iloc[i:i + args.batch_size].copy()
        n = len(batch_df)
        print(f'\n[Batch {batch_idx + 1}/{n_batches}] rows {i}-{i + n} | '
              f'overall {done}/{total} ({100 * done / total:.1f}%) done', flush=True)
        infer_requests = [
            InferRequest(messages=[get_message('image', p, args.prompt)])
            for p in batch_df['image_path'].tolist()
        ]

        metric.reset()  # time only this batch's inference
        responses = infer_batch(engine, infer_requests, request_config, metric)
        stats = metric.compute()

        batch_df['query'] = args.prompt
        batch_df['response'] = responses
        buffer.append(batch_df)
        done += n

        rt = max(stats.get('runtime', 1e-9), 1e-9)
        img_s = n / rt
        eta_min = (total - done) / img_s / 60 if img_s > 0 else float('inf')
        print(f'[Batch {batch_idx + 1}/{n_batches}] {n} imgs in {rt:.1f}s | '
              f'{img_s:.2f} img/s | in {stats.get("num_prompt_tokens", 0) / rt:.0f} tok/s | '
              f'out {stats.get("num_generated_tokens", 0) / rt:.0f} tok/s | ETA {eta_min:.1f} min',
              flush=True)

        # Flush every N batches, and always on the final batch.
        is_last = (i + args.batch_size) >= total
        if (batch_idx + 1) % args.save_every_n_batches == 0 or is_last:
            print(f'Saving after batch {batch_idx + 1} ({len(buffer)} batches buffered)')
            save_buffer(buffer, output_dir, args.prefix)
            buffer.clear()

    print('--- All batches complete ---')
