# Copyright (c) Alibaba, Inc. and its affiliates.
"""Resumable IN-PROCESS inference for the 3D CT-volume checkpoint over a parquet/CSV.

Unlike ../sglang/sglang_api_infer.py (which POSTs base64 2D images to a hosted sglang
server), CT volumes ride the model's ``<video>`` channel and are routed to the trained
3D encoder (model.visual_3d) -- a path sglang/vLLM don't serve. So this loads the
checkpoint locally with the exact recipe from notebooks/model_loading.ipynb:

    args  = InferArguments(model=CKPT, max_new_tokens=...)      # restores template/system/dtype
    model, template = prepare_model_template(args)              # attaches model.visual_3d
    engine = TransformersEngine(model, template=template, max_batch_size=...)

Each dataset row supplies a CT volume PATH (local, or s3://rnd-data-lake/...safetensors,
or .nii(.gz)/.npy/DICOM-dir). The path column is chosen with --image-column (default
``volume_uri``, the ct_rate_eval.parquet column). The ``<video>`` placeholder is appended
to the prompt automatically, mirroring the training data.

Saving / nomenclature / columns / resume all match the sglang script:
  * outputs go to  <output-dir>/task_{TASK}_temperature_{TEMP}_dataset_{NAME}/
  * one non-overwriting parquet per flush: ``<prefix>_<slug>_<timestamp>.parquet``
  * columns = every input column + ``query`` + ``response``
  * resume skips volume paths already present in the output dir
Only successful rows are persisted, so failed rows are naturally retried on a rerun.

Example:
    python ct_volume_infer.py \
        --dataset /cache/fast_data_nas71/janhavi/data/ct_eval/ct_rate_eval.parquet \
        --image-column volume_uri \
        --prompt "Generate the report for the given CT volume." \
        --model /cache/.../checkpoint-26000 \
        --output-dir /cache/.../vlm_results/ct_rate_eval \
        --task autoreport --dataset-name ctrate --temperature 0.0
"""
import argparse
import json
import os
import secrets
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import pandas as pd
from tqdm import tqdm

# Make the local swift fork importable (this file lives in <swift_repo>/examples/infer/ct_volume/),
# mirroring the sys.path insert in notebooks/model_loading.ipynb.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

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


def load_processed_paths(output_dir: Path, prefix: str, image_column: str) -> Set[str]:
    """Scan prior parquets so a resumed run skips already-done volumes."""
    if not output_dir.exists():
        return set()
    parquet_files = sorted(output_dir.glob(f'{prefix}_*.parquet'))
    if not parquet_files:
        return set()
    processed: Set[str] = set()
    total_rows = 0
    for fp in parquet_files:
        try:
            batch_df = pd.read_parquet(fp, columns=[image_column])
        except Exception as e:  # noqa: BLE001
            print(f'Warning: could not read {fp} for resume: {e}')
            continue
        total_rows += len(batch_df)
        processed.update(batch_df[image_column].tolist())
    print(f'Resume: {total_rows} rows across {len(parquet_files)} files, '
          f'{len(processed)} unique volumes already done.')
    return processed


def save_buffer(rows: List[Dict[str, Any]], output_dir: Path, prefix: str) -> None:
    """Persist accumulated result rows to a single new parquet (never overwrites)."""
    if not rows:
        return
    out_df = pd.DataFrame(rows)
    slug = random_word_slug()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path = output_dir / f'{prefix}_{slug}_{timestamp}.parquet'
    out_df.to_parquet(path, engine='pyarrow', compression='snappy', index=False)
    print(f'Persisted {len(out_df)} rows -> {path}')


def load_input_dataframe(path: str, image_column: str) -> pd.DataFrame:
    """Load the eval set into a DataFrame keyed by the CT-volume path (`image_column`).

    Supports:
      * .parquet / .csv  -> read as-is (must already contain `image_column`).
      * .json / .jsonl   -> ShareGPT-style records ({"videos": [...], "messages": [{"from","value"}]}),
        as produced for the 13-Aug test sets. We take the CT volume (videos[0]) into `image_column` to
        run inference on, and keep the record's `messages` verbatim (as a JSON string, for parquet safety)
        so the reference conversation is saved alongside the model's output. The prompt actually sent to
        the model is the chosen registry --prompt, not the row's own message.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext not in ('.json', '.jsonl'):
        return pd.read_csv(path) if ext == '.csv' else pd.read_parquet(path)

    # These files may be a JSON array, a single object, or JSON-Lines (one object per line) -- the
    # 13-Aug test sets are JSONL despite the .json extension. Try whole-file JSON first, fall back to JSONL.
    with open(path, encoding='utf-8') as fh:
        text = fh.read()
    try:
        records = json.loads(text)
        if isinstance(records, dict):
            records = [records]
    except json.JSONDecodeError:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    rows: List[Dict[str, Any]] = []
    for rec in records:
        videos = rec.get('videos')
        # `videos` may be a bare path string OR a list of paths -> take the first volume either way.
        if isinstance(videos, str):
            vol = videos
        elif isinstance(videos, (list, tuple)) and videos:
            vol = videos[0]
        else:
            vol = None
        rows.append({
            image_column: vol,
            'messages': json.dumps(rec.get('messages'), ensure_ascii=False),  # kept for reference
        })
    return pd.DataFrame(rows)


def build_request(row: Dict[str, Any], args, InferRequest):
    """One InferRequest: system + '<prompt> <video>' with the volume on the videos channel."""
    messages = [
        {'role': 'system', 'content': args.system},
        {'role': 'user', 'content': f'{args.prompt} <video>'},
    ]
    return InferRequest(messages=messages, videos=[row[args.image_column]])


def main(args) -> None:
    # --- Pick GPUs BEFORE importing torch/swift (mirrors the notebook cell 0) ------
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    if args.vision_3d_debug:
        os.environ['VISION_3D_DEBUG'] = '1'

    from swift.arguments import InferArguments
    from swift.pipelines.utils import prepare_model_template
    from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig

    run_subdir = f'task_{args.task}_temperature_{args.temperature}_dataset_{args.dataset_name}'
    output_dir = Path(args.output_dir).resolve() / run_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f'Output dir: {output_dir}')

    # --- Load & prepare dataset (parquet/csv, or ShareGPT .json/.jsonl test sets) ----------------
    df = load_input_dataframe(args.dataset, args.image_column)
    if args.image_column not in df.columns:
        raise ValueError(f'--image-column {args.image_column!r} not in dataset columns: {list(df.columns)}')
    df = df[df[args.image_column].notna()].reset_index(drop=True)
    df = df.drop_duplicates(subset=[args.image_column]).reset_index(drop=True)
    print(f'Dataset {args.dataset} loaded: {len(df)} unique rows (key column: {args.image_column})')

    # Data-parallel sharding: each process owns a fixed, disjoint stripe of the dataset
    # (rows i where i % num_shards == shard_id). Deterministic, so shards never collide even
    # though they share one OUTPUT_DIR. Run one process per GPU to use all GPUs at once.
    if args.num_shards > 1:
        df = df.iloc[args.shard_id::args.num_shards].reset_index(drop=True)
        print(f'Shard {args.shard_id}/{args.num_shards}: {len(df)} rows assigned to this process.')

    processed = load_processed_paths(output_dir, args.prefix, args.image_column)
    if processed:
        before = len(df)
        df = df[~df[args.image_column].isin(processed)].reset_index(drop=True)
        print(f'Skipping {before - len(df)} already-processed rows; {len(df)} remain.')

    if args.limit > 0:
        df = df.head(args.limit).reset_index(drop=True)
        print(f'Limit enabled: processing first {len(df)} rows.')

    if df.empty:
        print('Nothing to do. Exiting.')
        return

    # --- Load the checkpoint (in-process; attaches the 3D CT encoder) -----------
    infer_args_kwargs = dict(model=args.model, max_new_tokens=args.max_tokens)
    if args.torch_dtype:
        infer_args_kwargs['torch_dtype'] = args.torch_dtype
    if args.attn_impl:
        infer_args_kwargs['attn_impl'] = args.attn_impl
    infer_args = InferArguments(**infer_args_kwargs)
    model, template = prepare_model_template(infer_args)
    engine = TransformersEngine(model, template=template, max_batch_size=args.max_batch_size)
    print(f'Model loaded from {args.model} | visual_3d attached: {hasattr(model, "visual_3d")}')
    print(f'concurrency(max_batch_size) {args.max_batch_size} | max_tokens {args.max_tokens} | '
          f'temperature {args.temperature} | top_p {args.top_p} | top_k {args.top_k}')

    request_config = RequestConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        # Sent explicitly: the checkpoint's generation_config may omit these; passing
        # them avoids sampler-verify edge cases (see the swift_july top_p/top_k note).
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )

    rows = df.to_dict('records')
    buffer: List[Dict[str, Any]] = []
    success = fail = 0

    def infer_one(row: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single row; never raises -- failures are recorded, not saved."""
        try:
            req = build_request(row, args, InferRequest)
            resp = engine.infer([req], request_config)
            content = resp[0].choices[0].message.content
            return {**row, 'query': args.prompt, 'response': content, 'status': 'success'}
        except Exception as e:  # noqa: BLE001
            return {**row, 'query': args.prompt, 'response': f'ERROR: {e}', 'status': 'failure'}

    def infer_minibatch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batched infer for max_batch_size>1; on any batch error, retry each item alone."""
        if len(batch) == 1:
            return [infer_one(batch[0])]
        try:
            reqs = [build_request(r, args, InferRequest) for r in batch]
            resps = engine.infer(reqs, request_config, use_tqdm=False)
            return [{**r, 'query': args.prompt, 'response': resp.choices[0].message.content,
                     'status': 'success'} for r, resp in zip(batch, resps)]
        except Exception:  # noqa: BLE001 -- isolate the bad volume, keep the rest
            return [infer_one(r) for r in batch]

    # ---- TWO-TURN inference (used when --prompt2 is set; e.g. cls -> report multiturn) -----------
    def _turn1_messages() -> List[Dict[str, str]]:
        return [{'role': 'system', 'content': args.system},
                {'role': 'user', 'content': f'{args.prompt} <video>'}]

    def infer_two_turn_one(row: Dict[str, Any]) -> Dict[str, Any]:
        """turn1 = --prompt (with the volume); turn2 = --prompt2 with the volume + turn-1 answer in context."""
        try:
            vol = row[args.image_column]
            m1 = _turn1_messages()
            r1 = engine.infer([InferRequest(messages=list(m1), videos=[vol])],
                              request_config)[0].choices[0].message.content
            m2 = list(m1) + [{'role': 'assistant', 'content': r1}, {'role': 'user', 'content': args.prompt2}]
            r2 = engine.infer([InferRequest(messages=m2, videos=[vol])],
                              request_config)[0].choices[0].message.content
            return {**row, 'query': args.prompt, 'query2': args.prompt2, 'response_turn1': r1,
                    'response_turn2': r2, 'response': r2, 'status': 'success'}
        except Exception as e:  # noqa: BLE001
            return {**row, 'query': args.prompt, 'query2': args.prompt2, 'response_turn1': None,
                    'response_turn2': None, 'response': f'ERROR: {e}', 'status': 'failure'}

    def infer_two_turn_minibatch(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Batched two-turn: run turn1 for the whole batch, then turn2 for the whole batch."""
        if len(batch) == 1:
            return [infer_two_turn_one(batch[0])]
        try:
            vols = [r[args.image_column] for r in batch]
            m1 = _turn1_messages()
            r1s = [x.choices[0].message.content for x in
                   engine.infer([InferRequest(messages=list(m1), videos=[v]) for v in vols],
                                request_config, use_tqdm=False)]
            reqs2 = [InferRequest(messages=list(m1) + [{'role': 'assistant', 'content': r1},
                                                       {'role': 'user', 'content': args.prompt2}], videos=[v])
                     for r1, v in zip(r1s, vols)]
            r2s = [x.choices[0].message.content for x in engine.infer(reqs2, request_config, use_tqdm=False)]
            return [{**r, 'query': args.prompt, 'query2': args.prompt2, 'response_turn1': r1,
                     'response_turn2': r2, 'response': r2, 'status': 'success'}
                    for r, r1, r2 in zip(batch, r1s, r2s)]
        except Exception:  # noqa: BLE001 -- isolate the bad volume, keep the rest
            return [infer_two_turn_one(r) for r in batch]

    process_minibatch = infer_two_turn_minibatch if args.prompt2 else infer_minibatch
    if args.prompt2:
        print(f'TWO-TURN mode: turn1={args.prompt!r} | turn2={args.prompt2!r}')

    try:
        with tqdm(total=len(rows), desc='Inferring', unit='vol') as pbar:
            for start in range(0, len(rows), args.max_batch_size):
                batch = rows[start:start + args.max_batch_size]
                for res in process_minibatch(batch):
                    if res.pop('status') == 'success':
                        buffer.append(res)
                        success += 1
                    else:
                        fail += 1
                    pbar.update(1)
                    pbar.set_postfix(ok=success, fail=fail)
                if len(buffer) >= args.save_every:
                    save_buffer(buffer, output_dir, args.prefix)
                    buffer.clear()
    finally:
        if buffer:
            save_buffer(buffer, output_dir, args.prefix)
            buffer.clear()

    print(f'--- Done. {success} succeeded, {fail} failed (failures not saved; rerun to retry). ---')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resumable in-process CT-volume inference over a parquet/CSV.')
    parser.add_argument('--dataset', type=str, required=True, help='Input CSV/Parquet dataset')
    parser.add_argument('--image-column', type=str, default='volume_uri',
                        help='Column holding the CT volume path (local / s3:// / .nii/.npy/DICOM-dir)')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt/question for the model (turn 1)')
    parser.add_argument('--prompt2', type=str, default=None,
                        help='Optional second-turn prompt. If set, runs TWO-TURN inference: turn1=--prompt '
                             '(with the volume), then turn2=--prompt2 with the volume + turn-1 answer in '
                             'context. Output adds response_turn1/response_turn2 (response = turn2).')
    parser.add_argument('--model', type=str, required=True, help='Checkpoint path (dir with args.json/config.json)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Base dir; a task_{TASK}_temperature_{TEMP}_dataset_{NAME} subdir is created under it')
    parser.add_argument('--task', type=str, default='autoreport', help='Task name, e.g. autoreport / tags')
    parser.add_argument('--dataset-name', type=str, default='ctrate', help='Short dataset identifier')
    parser.add_argument('--prefix', type=str, default='ct_infer', help='Filename prefix for output parquets')
    parser.add_argument('--system', type=str, default='You are a helpful medical assistant.',
                        help='System prompt (checkpoint was trained with this one)')
    # generation / engine knobs
    parser.add_argument('--max-batch-size', type=int, default=1, help='Volumes per forward pass (VRAM-bound)')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Max new tokens')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=0.8, help='Nucleus sampling (sent explicitly)')
    parser.add_argument('--top-k', type=int, default=20, help='Top-k sampling (sent explicitly)')
    parser.add_argument('--seed', type=int, default=None, help='Optional sampling seed for reproducibility')
    parser.add_argument('--torch-dtype', type=str, default=None,
                        help='Override dtype, e.g. bfloat16 (checkpoint default is float32); ~2x faster + fits 1 GPU')
    parser.add_argument('--attn-impl', type=str, default=None,
                        help="Attention impl: sdpa (safe default) / flash_attn (faster, but can error on the "
                             "CT path -- fall back to sdpa if it does) / eager")
    parser.add_argument('--num-shards', type=int, default=1,
                        help='Total data-parallel shards (run one process per GPU with the same value)')
    parser.add_argument('--shard-id', type=int, default=0, help='This process shard index in [0, num_shards)')
    parser.add_argument('--gpus', type=str, default='0,1,2,3',
                        help='CUDA_VISIBLE_DEVICES, set before importing torch. For data-parallel pass ONE gpu '
                             'per process (e.g. --gpus 0 --shard-id 0). Empty = leave env as-is')
    parser.add_argument('--vision-3d-debug', action='store_true', help='Set VISION_3D_DEBUG=1 for encoder logs')
    parser.add_argument('--save-every', type=int, default=500, help='Flush a parquet every N successful rows')
    parser.add_argument('--limit', type=int, default=0, help='Process at most N rows (0 = all)')
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print('\nInterrupted by user.')
