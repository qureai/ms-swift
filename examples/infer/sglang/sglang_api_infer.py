# Copyright (c) Alibaba, Inc. and its affiliates.
"""Resumable async VLM inference against a *hosted* sglang OpenAI-compatible API.

Unlike ``run_sglang_inference_resumable.py`` (which builds an in-process
SglangEngine), this talks to a server already started with ``swift deploy
--infer_backend sglang`` (see ../deploy/sglang.sh). After the server prints
``Uvicorn running on http://0.0.0.0:8000``, point this client at it.

Saving / nomenclature / columns / resume all match run_sglang_inference_resumable.py:
  * outputs go to  <output-dir>/task_{TASK}_temperature_{TEMP}_dataset_{NAME}/
  * one non-overwriting parquet per flush: ``<prefix>_<slug>_<timestamp>.parquet``
  * columns = every input column + ``query`` + ``response``
  * resume skips image_paths already present in the output dir
Only successful rows are persisted, so failed requests are naturally retried on rerun.

Images are sent as base64 data URIs; ``--max-pixels`` downscales client-side before
encoding (caps vision-token cost without restarting the server). The server-side cap
is MAX_PIXELS in ../deploy/sglang.sh; the smaller of the two wins.

Example:
    python sglang_api_infer.py \
        --dataset /cache/.../manifest.csv \
        --prompt "Generate a detailed ontology report ..." \
        --output-dir /cache/.../vlm_results/my_run \
        --task autoreporting --dataset-name internal --temperature 0.5 \
        --model Qwen2.5_8B --base-url http://localhost:8000/v1 \
        --concurrency 64 --max-pixels 2250000
"""
import argparse
import asyncio
import base64
import io
import os
import secrets
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Set

import openai
import pandas as pd
from PIL import Image
from tqdm import tqdm

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


def load_processed_image_paths(output_dir: Path, prefix: str) -> Set[str]:
    """Scan prior parquets so a resumed run skips already-done images."""
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


def encode_image(image_path: str, max_pixels: int) -> str:
    """Load, optionally downscale to <= max_pixels, and return a base64 data URI."""
    img = Image.open(image_path).convert('RGB')
    if max_pixels and img.width * img.height > max_pixels:
        scale = (max_pixels / (img.width * img.height)) ** 0.5
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


async def call_api(client: 'openai.AsyncOpenAI', data_uri: str, args) -> str:
    """One chat-completion call with retries/backoff. Returns the response text."""
    last_err = None
    for attempt in range(1, args.max_retries + 1):
        try:
            resp = await client.chat.completions.create(
                model=args.model,
                messages=[{
                    'role': 'user',
                    'content': [
                        {'type': 'image_url', 'image_url': {'url': data_uri}},
                        {'type': 'text', 'text': args.prompt},
                    ],
                }],
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                # top_p/top_k MUST be sent: this checkpoint's generation_config.json
                # omits them, so swift falls back to None and sglang's verify() crashes
                # with "'<' not supported between float and NoneType". See --top-p/--top-k.
                top_p=args.top_p,
                extra_body={'top_k': args.top_k},
            )
            return resp.choices[0].message.content
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < args.max_retries:
                await asyncio.sleep(args.retry_delay * attempt)
    raise last_err


async def process_row(row: Dict[str, Any], client, sem, executor, args) -> Dict[str, Any]:
    """Encode the image (in a thread) and call the API for one dataset row."""
    async with sem:
        try:
            loop = asyncio.get_running_loop()
            data_uri = await loop.run_in_executor(executor, encode_image, row['image_path'], args.max_pixels)
            response = await call_api(client, data_uri, args)
            return {**row, 'query': args.prompt, 'response': response, 'status': 'success'}
        except Exception as e:  # noqa: BLE001
            return {**row, 'query': args.prompt, 'response': f'ERROR: {e}', 'status': 'failure'}


async def main(args) -> None:
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
        return

    print(f'Endpoint {args.base_url} | model {args.model} | concurrency {args.concurrency} | '
          f'max_pixels {args.max_pixels} | max_tokens {args.max_tokens} | temperature {args.temperature}')

    client = openai.AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    sem = asyncio.Semaphore(args.concurrency)
    executor = ThreadPoolExecutor(max_workers=args.encode_workers)

    rows = df.to_dict('records')
    tasks = [process_row(r, client, sem, executor, args) for r in rows]

    buffer: List[Dict[str, Any]] = []
    success = fail = 0
    try:
        with tqdm(total=len(tasks), desc='Inferring', unit='img') as pbar:
            for fut in asyncio.as_completed(tasks):
                res = await fut
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
        executor.shutdown(wait=True)
        await client.close()

    print(f'--- Done. {success} succeeded, {fail} failed (failures not saved; rerun to retry). ---')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Resumable async VLM inference against a hosted sglang API.')
    parser.add_argument('--dataset', type=str, required=True, help='Input CSV/Parquet dataset')
    parser.add_argument('--prompt', type=str, required=True, help='Prompt/question for the model')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Base directory; a task_{TASK}_temperature_{TEMP}_dataset_{NAME} subdir is created under it')
    parser.add_argument('--task', type=str, required=True, help='Task name, e.g. bbox / asp / autoreporting')
    parser.add_argument('--dataset-name', type=str, required=True, help='Short dataset identifier, e.g. internal')
    parser.add_argument('--prefix', type=str, default='vlm_results', help='Filename prefix for output parquets')
    # server / client
    parser.add_argument('--base-url', type=str, default='http://localhost:8000/v1', help='sglang OpenAI endpoint')
    parser.add_argument('--model', type=str, default='Qwen2.5_8B', help='served_model_name from the deploy')
    parser.add_argument('--api-key', type=str, default='EMPTY', help='Dummy key (sglang ignores it)')
    parser.add_argument('--concurrency', type=int, default=64, help='Max in-flight requests')
    parser.add_argument('--encode-workers', type=int, default=16, help='Threads for image load/resize/base64')
    parser.add_argument('--max-pixels', type=int, default=1500 * 1500,
                        help='Downscale images so w*h <= this before sending (0 = no client-side resize)')
    parser.add_argument('--max-tokens', type=int, default=8000, help='Max new tokens (<= server max_new_tokens)')
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--top-p', type=float, default=0.8,
                        help='Nucleus sampling; must be sent because the checkpoint gen config omits it')
    parser.add_argument('--top-k', type=int, default=20,
                        help='Top-k sampling; must be sent because the checkpoint gen config omits it')
    parser.add_argument('--max-retries', type=int, default=5, help='Retries per request on failure')
    parser.add_argument('--retry-delay', type=float, default=3.0, help='Base backoff seconds (multiplied by attempt)')
    parser.add_argument('--save-every', type=int, default=10000, help='Flush a parquet every N successful rows')
    parser.add_argument('--limit', type=int, default=0, help='Process at most N rows (0 = all)')
    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print('\nInterrupted by user.')
