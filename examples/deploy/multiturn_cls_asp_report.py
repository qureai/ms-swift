#!/usr/bin/env python
"""Multi-turn inference: cls-JSON -> ASP -> ontology report, in ONE growing conversation.

    python examples/deploy/multiturn_cls_asp_report.py \
        --ckpt   <checkpoint dir> \
        --input  <jsonl with {"images": [...]} per line> \
        --out    <result.jsonl>

WHY NOT `swift infer`
`swift infer` runs each row as a fixed, pre-written conversation: it cannot feed turn N+1 the
model's OWN turn-N output, because the assistant turns are supplied by the dataset. This task
needs exactly that -- the ASP reasoning is conditioned on the classification the model just
produced, and the report on both. So we drive the engine turn by turn and append each
generated answer to `messages` before asking the next question. This is real multi-turn
inference, not three independent prompts.

BACKEND: PtEngine (transformers). vLLM/sglang would be faster per call but their batch API is
built around the one-shot conversation; the turn-by-turn append is cleanest on the pt engine,
and at 100 cases x 3 turns the throughput is fine. The ML-Decoder head is NOT loaded here --
this script is about the generated text, not the classifier logits (for those, see
mldecoder_infer_plugin.py, MODE=head).

The three prompts are copied VERBATIM from the training file
`06_Aug_2026_cls_json_multiturn_asp_graph_report.json` (verified byte-identical), so the model
is asked exactly what it was trained on.
"""
import argparse
import json
import os
import sys
import time
from dataclasses import replace

# the qure plugins are restored from the checkpoint's args.json; force inference mode so their
# train-only patches (loss hooks, augmentation, ce-weight) are not installed
os.environ.setdefault('QURE_PLUGIN_MODE', 'infer')
# match the training/eval image budget: 2500 vision tokens/image
os.environ.setdefault('IMAGE_MAX_TOKEN_NUM', '2500')

import torch

SWIFT_ROOT = '/fastdata4/sahil_work/VLM_qure/vlm/swift'
if SWIFT_ROOT not in sys.path:
    sys.path.insert(0, SWIFT_ROOT)

# This fork keeps the engines under swift.infer_engine (not swift.llm), and its pt backend is
# TransformersEngine, not PtEngine.
from swift.infer_engine import InferRequest, RequestConfig, TransformersEngine  # noqa: E402

# --- the three turns, verbatim from the training data (sha-verified) -------------------------
CLS = ('Classify the abnormalities in the given chest x-ray as a detailed JSON report. Return '
       'a single JSON object with the ontology categories general, lung_cancer, copd, ild, '
       'pulmonary_hypertension, edema and other_findings. Under each category, report every '
       'finding of that category as an object with the keys presence, size, location and '
       'characteristics. Set presence to true when the finding is present in the image and '
       'false when it is absent, and report every finding including the absent ones. For an '
       'absent finding set size, location and characteristics to null; for a present finding '
       'give a very short characterization in each, or null where it does not apply.')
ASP = ('Generate Answer Set Programming (ASP)-based reasoning for the provided medical image. '
       'Include multi-level diagnostic analysis and structured, step-wise clinical reasoning. '
       'Present the logic in a transparent, explainable format appropriate for computational '
       'reasoning frameworks.')
REPORT = ('Generate a detailed ontology report of the given X-ray with Findings, Summary, '
          'Complete Reasoning, Impression, Recommendations, and Prominence Score.')
# Per-turn token cap, each set just above that turn's TRAINING-target maximum so no legitimate
# output is ever truncated, but a degenerate generation cannot grind to a 28000 global cap.
# This matters because in pt-batched decode there is no early eviction: ONE runaway case
# (cls OR asp -- both were observed hitting 28000 and pinning 80 GB of KV) blocks its entire
# batch, which is what repeatedly left shards stuck for 800s+. Training maxima: cls ~6k,
# asp ~10.2k, report ~1k tokens; caps are set with headroom above those.
#   The right long-term fix is sglang continuous batching (evicts finished seqs, per-request
#   caps), not this -- these caps are the pragmatic guard for the pt path.
TURN_CAP = {'cls': 8000, 'asp': 12000, 'report': 4000}
TURNS = [('cls', CLS), ('asp', ASP), ('report', REPORT)]

SYSTEM = 'You are a helpful medical assistant.'


def load_rows(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            imgs = d.get('images') or ([d['image']] if 'image' in d else [])
            imgs = [x['path'] if isinstance(x, dict) else x for x in imgs]
            if imgs:
                rows.append({'images': imgs, 'meta': d})
    return rows


def run_batch(engine, batch, req_cfg):
    """Drive N cases through the 3 turns TURN-BY-TURN, batched.

    Instead of finishing one case's 3 turns before starting the next (which decodes a single
    sequence at a time and leaves a 143 GB GPU at ~27%), we run turn 1 for ALL N cases at once,
    then turn 2 for all N (each conditioned on ITS OWN turn-1 output), then turn 3. The engine
    batches the N requests in one forward, so utilisation scales with N while the multi-turn
    conditioning is unchanged -- case k's turn t still sees only case k's earlier answers.

    `batch` is a list of {'images': [...]}. Returns a list of {name: {text, tokens}} per case.
    """
    n = len(batch)
    # one growing message list per case; image only in the first user turn
    msgs = [[{'role': 'system', 'content': SYSTEM}] for _ in range(n)]
    out = [{} for _ in range(n)]
    for i, (name, prompt) in enumerate(TURNS):
        reqs = []
        for k in range(n):
            content = f'{prompt}\n<image>' if i == 0 else prompt
            msgs[k].append({'role': 'user', 'content': content})
            reqs.append(InferRequest(messages=msgs[k], images=batch[k]['images'],
                                     chat_template_kwargs={'enable_thinking': False}))
        # per-turn cap so a runaway ASP can't hold up the whole batch
        turn_cfg = replace(req_cfg, max_tokens=min(req_cfg.max_tokens, TURN_CAP.get(name, req_cfg.max_tokens)))
        t0 = time.time()
        resps = engine.infer(reqs, turn_cfg, use_tqdm=False)
        dt = time.time() - t0
        for k, resp in enumerate(resps):
            text = resp.choices[0].message.content
            out[k][name] = {'text': text,
                            'tokens': resp.usage.completion_tokens if resp.usage else None}
            msgs[k].append({'role': 'assistant', 'content': text})
        toks = sum((r.usage.completion_tokens if r.usage else 0) for r in resps)
        print(f'  turn {name}: {n} cases · {dt:.0f}s · {toks} tok · {toks/max(dt,1):.0f} tok/s '
              f'(batched)', flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--max-new-tokens', type=int, default=28000)
    ap.add_argument('--max-len', type=int, default=28000)
    ap.add_argument('--shard', type=int, default=0, help='this shard index (0-based)')
    ap.add_argument('--num-shards', type=int, default=1,
                    help='total shards; case i goes to shard i %% num_shards')
    ap.add_argument('--batch', type=int, default=16,
                    help='cases decoded together per turn (fills the GPU). A whole shard fits '
                         'in one batch at these sizes.')
    args = ap.parse_args()

    rows = load_rows(args.input)
    if args.limit:
        rows = rows[:args.limit]
    # deterministic round-robin shard so N GPUs cover disjoint cases with no coordination
    if args.num_shards > 1:
        rows = [r for i, r in enumerate(rows) if i % args.num_shards == args.shard]
    print(f'{len(rows)} cases (shard {args.shard}/{args.num_shards}) · ckpt {args.ckpt}',
          flush=True)

    # sdpa, not flash_attention_2: on this fork's pt engine flash-attn triggers a
    # `CUDA error: illegal memory access` on the vision tower mid-generation. sdpa is a touch
    # slower but stable.
    # max_batch_size = args.batch: the whole point of batching cases per turn is to fill the
    # GPU. At 1 the engine decodes one sequence at a time and a 143 GB card sits at ~27%.
    engine = TransformersEngine(args.ckpt, max_batch_size=args.batch,
                                torch_dtype=torch.bfloat16, attn_impl='sdpa', use_hf=True,
                                max_model_len=args.max_len)
    # THINKING MUST BE OFF. The model was trained with `--add_non_thinking_prefix true
    # --loss_scale ignore_empty_think`, i.e. every assistant turn is an empty <think></think>
    # then content. Left in its default thinking mode the model instead emits a full <think>
    # chain on the ASP turn and degenerates into a repetition loop that runs to the 28000-token
    # cap (observed: asp=28000 tok, 97% repeated lines, 562s/case). Setting these on the
    # template reproduces the training/eval config and makes every turn terminate.
    engine.template.enable_thinking = False
    engine.template.add_non_thinking_prefix = True
    req_cfg = RequestConfig(max_tokens=args.max_new_tokens, temperature=0.0,
                            top_k=1, top_p=1.0, repetition_penalty=1.0)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t_start = time.time()
    done = 0
    with open(args.out, 'w') as fo:
        for b0 in range(0, len(rows), args.batch):
            batch = rows[b0:b0 + args.batch]
            t0 = time.time()
            try:
                results = run_batch(engine, batch, req_cfg)
                errs = [None] * len(batch)
            except Exception as e:
                # a whole-batch failure is logged per case so nothing is silently dropped
                results = [{} for _ in batch]
                errs = [f'{type(e).__name__}: {e}'] * len(batch)
            for row, turns, err in zip(batch, results, errs):
                fo.write(json.dumps({'images': row['images'], 'meta': row['meta'],
                                     'turns': turns, 'error': err}) + '\n')
            fo.flush()
            done += len(batch)
            dt = time.time() - t_start
            eta = dt / done * (len(rows) - done)
            print(f'[batch {done}/{len(rows)}] {time.time()-t0:.0f}s · '
                  f'{(time.time()-t0)/len(batch):.1f}s/case · ETA {eta/60:.0f}m', flush=True)

    print(f'\ndone -> {args.out}  ({len(rows)} cases in {(time.time()-t_start)/60:.1f}m)',
          flush=True)


if __name__ == '__main__':
    main()
