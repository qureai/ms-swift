#!/usr/bin/env python
"""Head-guided 3-stage inference: ML-Decoder positives -> prompt -> JSON -> ASP -> report.

Pipeline
  1. The ML-Decoder head's positive findings (thresholded at the per-tag sens==spec / EER
     point) are read from --positives (key -> [finding,...], most-confident first).
  2. Those positives are injected into the stage-1 (classification) user prompt as a prior.
  3. Three DEPENDENT sglang passes generate, carrying the conversation forward each time:
       stage 1  cls JSON        (prompt has the head positives)
       stage 2  ASP reasoning   (sees the JSON it just produced)
       stage 3  ontology report (sees the JSON + ASP)
  Each pass is a full `swift infer` (sglang) run; between passes we append the model's own
  previous answer as an assistant turn and add the next task's user turn.

Usage
  python infer_headguided_3stage.py --ckpt <dir> --input <studies.jsonl> \
      --positives <head_positives_eer.json> --out_dir <dir> [--limit N] [--dp 8]

Output: <out_dir>/final.jsonl  ->  {key,image,head_positives,cls_json,asp,report}
"""
import argparse
import json
import os
import subprocess
import sys

SWIFT_ROOT = '/fastdata4/sahil_work/VLM_qure/vlm/swift'
FM_REPO = os.environ.get('FM_ACCELERATE_REPO',
                         '/fastdata4/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate')
CONDA_ENV = os.environ.get('CONDA_ENV', '/home/ubuntu/miniconda3/envs/ms-swift-may-s4')
SYSTEM = 'You are a helpful medical assistant.'

CLS_PROMPT = ('Classify the abnormalities in the given chest x-ray as a detailed JSON report. '
              'Return a single JSON object with the ontology categories general, lung_cancer, copd, '
              'ild, pulmonary_hypertension, edema and other_findings. Under each category, report '
              'every finding of that category as an object with the keys presence, size, location and '
              'characteristics. Set presence to true when the finding is present in the image and '
              'false when it is absent, and report every finding including the absent ones. For an '
              'absent finding set size, location and characteristics to null; for a present finding '
              'give a very short characterization in each, or null where it does not apply.')
ASP_PROMPT = ('Generate Answer Set Programming (ASP)-based reasoning for the provided medical image. '
              'Include multi-level diagnostic analysis and structured, step-wise clinical reasoning. '
              'Present the logic in a transparent, explainable format appropriate for computational '
              'reasoning frameworks.')
REPORT_PROMPT = ('Generate a detailed ontology report of the given X-ray with Findings, Summary, '
                 'Complete Reasoning, Impression, Recommendations, and Prominence Score.')


def base(p):
    p = p['path'] if isinstance(p, dict) else p
    return os.path.basename(str(p)).rstrip("'}").rstrip('"')


def inject(positives):
    """Build the stage-1 user text with the head's positive findings as a prior."""
    if positives:
        lst = ', '.join(positives)
        prior = ('\n\nA preliminary automated image analysis flags the following findings as likely '
                 f'PRESENT (ranked by confidence): {lst}. Treat this as a prior: verify each against '
                 'what you actually see in the image, confirming or rejecting it, and still report '
                 'every other finding as usual.')
    else:
        prior = ('\n\nA preliminary automated image analysis flags no findings as clearly present; '
                 'read the image carefully regardless.')
    return CLS_PROMPT + prior + '\n<image>'


def run_swift(prepared, result_path, out_dir, dp, max_new_tokens, log):
    env = dict(os.environ)
    env['PATH'] = f"{CONDA_ENV}/bin:" + env.get('PATH', '')
    env['PYTHONPATH'] = f"{SWIFT_ROOT}:{FM_REPO}:" + env.get('PYTHONPATH', '')
    env.setdefault('IMAGE_MAX_TOKEN_NUM', '2500')
    cmd = [
        'swift', 'infer', '--model', env['CKPT'],
        '--val_dataset', prepared, '--result_path', result_path,
        '--max_new_tokens', str(max_new_tokens), '--max_length', '28000',
        '--temperature', '0', '--top_p', '1.0', '--top_k', '1',
        '--repetition_penalty', '1.0', '--enable_thinking', 'false',
        '--system', SYSTEM, '--use_hf', 'true', '--val_dataset_shuffle', 'false',
        '--write_batch_size', '200',
        '--infer_backend', 'sglang', '--sglang_dp_size', str(dp), '--sglang_tp_size', '1',
        '--sglang_mem_fraction_static', '0.9', '--sglang_context_length', '28000',
    ]
    with open(log, 'w') as lf:
        r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f'stage failed (see {log})')


def read_results(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        img = (d.get('images') or [None])[0]
        if img:
            out[base(img)] = d.get('response', '')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--input', required=True, help='jsonl with images/image per row')
    ap.add_argument('--positives', required=True, help='key -> [finding,...] json from the head')
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--dp', type=int, default=8)
    ap.add_argument('--only-scored', action='store_true',
                    help='keep only studies that have a head-positives entry')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    os.environ['CKPT'] = a.ckpt
    positives = {base(k): v for k, v in json.load(open(a.positives)).items()}

    # -- collect studies --------------------------------------------------------------------
    studies = []
    for line in open(a.input):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        imgs = d.get('images') or ([d['image']] if 'image' in d else [])
        imgs = [x['path'] if isinstance(x, dict) else x for x in imgs]
        if not imgs:
            continue
        k = base(imgs[0])
        if a.only_scored and k not in positives:
            continue
        studies.append({'key': k, 'image': imgs[0], 'pos': positives.get(k, [])})
        if a.limit and len(studies) >= a.limit:
            break
    print(f'{len(studies)} studies', flush=True)

    # -- STAGE 1: cls JSON (head positives injected) ----------------------------------------
    s1_in = f'{a.out_dir}/s1_input.jsonl'
    with open(s1_in, 'w') as f:
        for s in studies:
            msgs = [{'role': 'user', 'content': inject(s['pos'])}]
            f.write(json.dumps({'messages': msgs, 'images': [s['image']]}) + '\n')
    print('stage 1 (cls JSON) ...', flush=True)
    run_swift(s1_in, f'{a.out_dir}/s1_result.jsonl', a.out_dir, a.dp, 8192, f'{a.out_dir}/s1.log')
    json_resp = read_results(f'{a.out_dir}/s1_result.jsonl')

    # -- STAGE 2: ASP (sees the JSON) -------------------------------------------------------
    s2_in = f'{a.out_dir}/s2_input.jsonl'
    with open(s2_in, 'w') as f:
        for s in studies:
            msgs = [{'role': 'user', 'content': inject(s['pos'])},
                    {'role': 'assistant', 'content': json_resp.get(s['key'], '')},
                    {'role': 'user', 'content': ASP_PROMPT}]
            f.write(json.dumps({'messages': msgs, 'images': [s['image']]}) + '\n')
    print('stage 2 (ASP) ...', flush=True)
    run_swift(s2_in, f'{a.out_dir}/s2_result.jsonl', a.out_dir, a.dp, 4096, f'{a.out_dir}/s2.log')
    asp_resp = read_results(f'{a.out_dir}/s2_result.jsonl')

    # -- STAGE 3: ontology report (sees JSON + ASP) -----------------------------------------
    s3_in = f'{a.out_dir}/s3_input.jsonl'
    with open(s3_in, 'w') as f:
        for s in studies:
            msgs = [{'role': 'user', 'content': inject(s['pos'])},
                    {'role': 'assistant', 'content': json_resp.get(s['key'], '')},
                    {'role': 'user', 'content': ASP_PROMPT},
                    {'role': 'assistant', 'content': asp_resp.get(s['key'], '')},
                    {'role': 'user', 'content': REPORT_PROMPT}]
            f.write(json.dumps({'messages': msgs, 'images': [s['image']]}) + '\n')
    print('stage 3 (report) ...', flush=True)
    run_swift(s3_in, f'{a.out_dir}/s3_result.jsonl', a.out_dir, a.dp, 4096, f'{a.out_dir}/s3.log')
    report_resp = read_results(f'{a.out_dir}/s3_result.jsonl')

    # -- merge ------------------------------------------------------------------------------
    final = f'{a.out_dir}/final.jsonl'
    with open(final, 'w') as f:
        for s in studies:
            f.write(json.dumps({
                'key': s['key'], 'image': s['image'], 'head_positives': s['pos'],
                'cls_json': json_resp.get(s['key'], ''), 'asp': asp_resp.get(s['key'], ''),
                'report': report_resp.get(s['key'], ''),
            }) + '\n')
    print(f'done -> {final} ({len(studies)} studies)', flush=True)


if __name__ == '__main__':
    main()
