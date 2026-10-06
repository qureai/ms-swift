#!/usr/bin/env python
"""FULL-140k stage-3 chain: cls-JSON -> ASP -> structured JSON report (template-Q, exact
training prompts), three dependent sglang passes at max throughput (DP=8).

Token budgets chosen so nothing is cut:
  turn1 cls   max_new  8192  (cls JSONs are ~4-5k tokens)
  turn2 ASP   max_new 12000  (ASPs are ~5-6k tokens; cap guards the turn-3 window)
  turn3 sjson max_new  6000  (reports are ~1.5-2k tokens)
  worst-case turn3 input ~= 5k cls + 2.5k image + 12k ASP + prompts < 21k; +6k < 28k ctx.

Output: <out>/final.jsonl {key,image,cls_json,asp,structured_report}
"""
import json
import os
import subprocess
import sys
import time

SWIFT_ROOT = '/fastdata4/sahil_work/VLM_qure/vlm/swift'
FM = '/fastdata4/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate'
CE = '/home/ubuntu/miniconda3/envs/ms-swift-may-s4'
CKPT = f'{SWIFT_ROOT}/output/Qwen3.5-2B_stage3_diverse_frozen_vit_head/v4-20260813-090333/checkpoint-300'
INPUT = f'{SWIFT_ROOT}/vlm_inference_analysis/results/joint_run/eval140k/input.jsonl'
OUT = f'{SWIFT_ROOT}/vlm_inference_analysis/results/joint_run/stage3_full140k_cls_asp_sjson'

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
SJ_PROMPT = 'Provide a structured JSON report for the X-ray.'


def base(p):
    p = p['path'] if isinstance(p, dict) else p
    return os.path.basename(str(p)).rstrip("'}").rstrip('"')


def run_swift(prepared, result, log, max_new):
    env = dict(os.environ)
    env['PATH'] = f'{CE}/bin:' + env.get('PATH', '')
    env['PYTHONPATH'] = f'{SWIFT_ROOT}:{FM}:' + env.get('PYTHONPATH', '')
    env.setdefault('IMAGE_MAX_TOKEN_NUM', '2500')
    cmd = ['swift', 'infer', '--model', CKPT,
           '--val_dataset', prepared, '--result_path', result,
           '--max_new_tokens', str(max_new), '--max_length', '28000',
           '--temperature', '0', '--top_p', '1.0', '--top_k', '1',
           '--repetition_penalty', '1.0', '--enable_thinking', 'false',
           '--system', 'You are a helpful medical assistant.', '--use_hf', 'true',
           '--val_dataset_shuffle', 'false', '--write_batch_size', '200',
           '--infer_backend', 'sglang', '--sglang_dp_size', '8', '--sglang_tp_size', '1',
           '--sglang_mem_fraction_static', '0.9', '--sglang_context_length', '28000']
    t0 = time.time()
    r = subprocess.run(cmd, env=env, stdout=open(log, 'w'), stderr=subprocess.STDOUT)
    print(f'  pass done in {(time.time()-t0)/3600:.2f} h (rc={r.returncode})', flush=True)
    if r.returncode != 0:
        sys.exit(f'swift infer failed: {log}')


def read_results(path):
    out = {}
    for line in open(path):
        try:
            d = json.loads(line)
        except Exception:
            continue
        img = (d.get('images') or [None])[0]
        if img:
            out[base(img)] = d.get('response', '')
    return out


os.makedirs(OUT, exist_ok=True)
cases = []
for line in open(INPUT):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line)
    imgs = d.get('images') or ([d['image']] if 'image' in d else [])
    imgs = [x['path'] if isinstance(x, dict) else x for x in imgs]
    if imgs:
        cases.append(imgs[0])
print(f'{len(cases):,} studies', flush=True)

# ---- turn 1: cls JSON --------------------------------------------------------------------
s1 = f'{OUT}/s1_input.jsonl'
with open(s1, 'w') as f:
    for p in cases:
        f.write(json.dumps({'messages': [{'role': 'user', 'content': CLS_PROMPT + '\n<image>'}],
                            'images': [p]}) + '\n')
print('turn 1/3: cls JSON ...', flush=True)
run_swift(s1, f'{OUT}/s1_result.jsonl', f'{OUT}/s1.log', 8192)
cls = read_results(f'{OUT}/s1_result.jsonl')

# ---- turn 2: ASP conditioned on cls ------------------------------------------------------
s2 = f'{OUT}/s2_input.jsonl'
with open(s2, 'w') as f:
    for p in cases:
        k = base(p)
        msgs = [{'role': 'user', 'content': CLS_PROMPT + '\n<image>'},
                {'role': 'assistant', 'content': cls.get(k, '')},
                {'role': 'user', 'content': ASP_PROMPT}]
        f.write(json.dumps({'messages': msgs, 'images': [p]}) + '\n')
print('turn 2/3: ASP ...', flush=True)
run_swift(s2, f'{OUT}/s2_result.jsonl', f'{OUT}/s2.log', 12000)
asp = read_results(f'{OUT}/s2_result.jsonl')

# ---- turn 3: structured JSON report conditioned on cls + ASP -----------------------------
s3 = f'{OUT}/s3_input.jsonl'
with open(s3, 'w') as f:
    for p in cases:
        k = base(p)
        msgs = [{'role': 'user', 'content': CLS_PROMPT + '\n<image>'},
                {'role': 'assistant', 'content': cls.get(k, '')},
                {'role': 'user', 'content': ASP_PROMPT},
                {'role': 'assistant', 'content': asp.get(k, '')},
                {'role': 'user', 'content': SJ_PROMPT}]
        f.write(json.dumps({'messages': msgs, 'images': [p]}) + '\n')
print('turn 3/3: structured JSON report ...', flush=True)
run_swift(s3, f'{OUT}/s3_result.jsonl', f'{OUT}/s3.log', 6000)
sj = read_results(f'{OUT}/s3_result.jsonl')

# ---- merge -------------------------------------------------------------------------------
with open(f'{OUT}/final.jsonl', 'w') as f:
    for p in cases:
        k = base(p)
        f.write(json.dumps({'key': k, 'image': p, 'cls_json': cls.get(k, ''),
                            'asp': asp.get(k, ''), 'structured_report': sj.get(k, '')}) + '\n')
print(f'ALL DONE -> {OUT}/final.jsonl ({len(cases):,} studies)', flush=True)
