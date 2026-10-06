#!/usr/bin/env python
"""Two-turn inference: cls-JSON classification, then structured JSON report — the stage-3
multiturn pattern (template P minus bbox / N chain), each turn a full sglang pass with the
model's own previous answer fed back.

    python infer_cls_then_sjson.py --ckpt <dir> --cases <jsonl> --out_dir <dir> [--dp 8]

cases.jsonl rows: {"images": [path], ...extra keys are carried through}
Output: <out_dir>/final.jsonl  {key,image,gt,cls_json,structured_report}
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

CLS_PROMPT = ('Classify the abnormalities in the given chest x-ray as a detailed JSON report. '
              'Return a single JSON object with the ontology categories general, lung_cancer, copd, '
              'ild, pulmonary_hypertension, edema and other_findings. Under each category, report '
              'every finding of that category as an object with the keys presence, size, location and '
              'characteristics. Set presence to true when the finding is present in the image and '
              'false when it is absent, and report every finding including the absent ones. For an '
              'absent finding set size, location and characteristics to null; for a present finding '
              'give a very short characterization in each, or null where it does not apply.')
SJSON_PROMPT = 'Provide a structured JSON report for the X-ray.'


def base(p):
    p = p['path'] if isinstance(p, dict) else p
    return os.path.basename(str(p)).rstrip("'}").rstrip('"')


def run_swift(ckpt, prepared, result_path, dp, max_new_tokens, log):
    env = dict(os.environ)
    env['PATH'] = f'{CONDA_ENV}/bin:' + env.get('PATH', '')
    env['PYTHONPATH'] = f'{SWIFT_ROOT}:{FM_REPO}:' + env.get('PYTHONPATH', '')
    env.setdefault('IMAGE_MAX_TOKEN_NUM', '2500')
    cmd = ['swift', 'infer', '--model', ckpt,
           '--val_dataset', prepared, '--result_path', result_path,
           '--max_new_tokens', str(max_new_tokens), '--max_length', '28000',
           '--temperature', '0', '--top_p', '1.0', '--top_k', '1',
           '--repetition_penalty', '1.0', '--enable_thinking', 'false',
           '--system', 'You are a helpful medical assistant.', '--use_hf', 'true',
           '--val_dataset_shuffle', 'false', '--write_batch_size', '100',
           '--infer_backend', 'sglang', '--sglang_dp_size', str(dp), '--sglang_tp_size', '1',
           '--sglang_mem_fraction_static', '0.9', '--sglang_context_length', '28000']
    with open(log, 'w') as lf:
        r = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        sys.exit(f'swift infer failed (see {log})')


def read_results(path):
    out = {}
    for line in open(path):
        d = json.loads(line)
        img = (d.get('images') or [None])[0]
        if img:
            out[base(img)] = d.get('response', '')
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--cases', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--dp', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    cases = [json.loads(l) for l in open(a.cases) if l.strip()]
    for c in cases:
        c['_key'] = base(c['images'][0])
    print(f'{len(cases)} cases', flush=True)

    # turn 1: cls JSON
    s1 = f'{a.out_dir}/s1_input.jsonl'
    with open(s1, 'w') as f:
        for c in cases:
            f.write(json.dumps({'messages': [{'role': 'user', 'content': CLS_PROMPT + '\n<image>'}],
                                'images': c['images']}) + '\n')
    print('turn 1: cls JSON ...', flush=True)
    run_swift(a.ckpt, s1, f'{a.out_dir}/s1_result.jsonl', a.dp, 8192, f'{a.out_dir}/s1.log')
    cls = read_results(f'{a.out_dir}/s1_result.jsonl')

    # turn 2: structured JSON report, conditioned on the model's own cls JSON
    s2 = f'{a.out_dir}/s2_input.jsonl'
    with open(s2, 'w') as f:
        for c in cases:
            msgs = [{'role': 'user', 'content': CLS_PROMPT + '\n<image>'},
                    {'role': 'assistant', 'content': cls.get(c['_key'], '')},
                    {'role': 'user', 'content': SJSON_PROMPT}]
            f.write(json.dumps({'messages': msgs, 'images': c['images']}) + '\n')
    print('turn 2: structured JSON report ...', flush=True)
    run_swift(a.ckpt, s2, f'{a.out_dir}/s2_result.jsonl', a.dp, 8192, f'{a.out_dir}/s2.log')
    sj = read_results(f'{a.out_dir}/s2_result.jsonl')

    final = f'{a.out_dir}/final.jsonl'
    with open(final, 'w') as f:
        for c in cases:
            f.write(json.dumps({'key': c['_key'], 'image': c['images'][0], 'gt': c.get('gt'),
                                'cls_json': cls.get(c['_key'], ''),
                                'structured_report': sj.get(c['_key'], '')}) + '\n')
    print(f'done -> {final}', flush=True)


if __name__ == '__main__':
    main()
