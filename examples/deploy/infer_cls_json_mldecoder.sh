#!/usr/bin/env bash
set -euo pipefail

# Inference for the joint cls-JSON + ML-Decoder checkpoint, using ms-swift only.
#
#   bash examples/deploy/infer_cls_json_mldecoder.sh <input.jsonl> <out_dir> [MODE]
#
# MODE=json   (default) generate the detailed classification JSON per image
# MODE=head   generate the JSON *and* dump the ML-Decoder head's 132 logits per image
# MODE=asp    cls JSON, then the ASP branch, as a 2-turn conversation
# MODE=graph  cls JSON, then the hierarchical graph branch
#
# <input.jsonl> may be either
#   {"images": ["/path/a.png"]}                                  <- prompt added for you
#   {"messages":[{"role":"user","content":"...\n<image>"}], "images":[...]}   <- used as-is
#
# BACKEND NOTE. MODE=json/asp/graph use sglang (fast). MODE=head forces --infer_backend pt,
# because vLLM/sglang reimplement the model and a forward hook on `model.visual` never fires
# there -- the head would silently produce nothing. pt is much slower, so only use MODE=head
# when you actually need the classifier outputs.
#
# PLUGIN NOTE. The head lives in the checkpoint as 25 `ml_decoder_head.*` tensors, but it is
# attached at RUNTIME. Without the plugin those keys are dropped silently and you get a random
# head with no error, so MODE=head must pass mldecoder_infer_plugin.py.

CONDA_ENV="${CONDA_ENV:-/home/ubuntu/miniconda3/envs/ms-swift-may-s4}"
export PATH="$CONDA_ENV/bin:$PATH"          # sglang JIT kernels need `ninja` on PATH
SWIFT_ROOT="/fastdata4/sahil_work/VLM_qure/vlm/swift"
FM_REPO="${FM_ACCELERATE_REPO:-/fastdata4/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate}"
export PYTHONPATH="$SWIFT_ROOT:$FM_REPO:${PYTHONPATH:-}"
export IMAGE_MAX_TOKEN_NUM="${IMAGE_MAX_TOKEN_NUM:-2500}"

CKPT="${CKPT:-$SWIFT_ROOT/output/Qwen3.5-2B_cls_json_mldecoder_joint/v9-20260807-004153/checkpoint-300}"
INPUT="${1:?usage: $0 <input.jsonl> <out_dir> [json|head|asp|graph]}"
OUT_DIR="${2:?usage: $0 <input.jsonl> <out_dir> [json|head|asp|graph]}"
MODE="${3:-json}"
mkdir -p "$OUT_DIR"

CLS_PROMPT='Classify the abnormalities in the given chest x-ray as a detailed JSON report. Return a single JSON object with the ontology categories general, lung_cancer, copd, ild, pulmonary_hypertension, edema and other_findings. Under each category, report every finding of that category as an object with the keys presence, size, location and characteristics. Set presence to true when the finding is present in the image and false when it is absent, and report every finding including the absent ones. For an absent finding set size, location and characteristics to null; for a present finding give a very short characterization in each, or null where it does not apply.'
ASP_PROMPT='Generate Answer Set Programming (ASP)-based reasoning for the provided medical image. Include multi-level diagnostic analysis and structured, step-wise clinical reasoning. Present the logic in a transparent, explainable format appropriate for computational reasoning frameworks.'
GRAPH_PROMPT='Generate a hierarchical positive findings graph and positive findings triples for the given image.'

[[ -d "$CKPT" ]] || { echo "checkpoint not found: $CKPT" >&2; exit 1; }
[[ -f "$INPUT" ]] || { echo "input not found: $INPUT" >&2; exit 1; }

# --- build the infer input: add the prompt(s) unless the row already has `messages` ----------
PREPARED="$OUT_DIR/infer_input.jsonl"
python3.12 - "$INPUT" "$PREPARED" "$MODE" "$CLS_PROMPT" "$ASP_PROMPT" "$GRAPH_PROMPT" <<'PY'
import json, sys
src, dst, mode, cls_p, asp_p, graph_p = sys.argv[1:7]
follow = {'asp': asp_p, 'graph': graph_p}.get(mode)
n = 0
with open(src) as f, open(dst, 'w') as o:
    for line in f:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        if 'messages' in d:                      # already a conversation, leave it alone
            o.write(json.dumps(d) + '\n'); n += 1; continue
        imgs = d.get('images') or ([d['image']] if 'image' in d else [])
        if not imgs:
            continue
        msgs = [{'role': 'user', 'content': f'{cls_p}\n<image>'}]
        if follow:
            # swift needs the assistant turn present for a multi-turn request; leave it empty
            # so generation continues from the follow-up question
            msgs += [{'role': 'assistant', 'content': ''}, {'role': 'user', 'content': follow}]
        o.write(json.dumps({'messages': msgs, 'images': imgs}) + '\n'); n += 1
print(f'prepared {n} records -> {dst}', file=sys.stderr)
PY

RESULT="$OUT_DIR/result.jsonl"
COMMON=(
    --model "$CKPT"
    --val_dataset "$PREPARED"
    --result_path "$RESULT"
    --max_new_tokens "${MAX_NEW_TOKENS:-8192}"
    --max_length "${MAX_LENGTH:-28000}"
    --temperature 0
    --top_p 1.0
    --top_k 1
    --repetition_penalty 1.0
    --enable_thinking false
    --system 'You are a helpful medical assistant.'
    --use_hf true
    --val_dataset_shuffle false
    --write_batch_size 200
)

if [[ "$MODE" == "head" ]]; then
    export MLD_CLS_OUT="$OUT_DIR/cls_logits.jsonl"
    echo "MODE=head -> pt backend, head logits to $MLD_CLS_OUT" >&2
    swift infer "${COMMON[@]}" \
        --infer_backend pt \
        --max_batch_size 1 \
        --external_plugins "$SWIFT_ROOT/swift_plugins/mldecoder_infer_plugin.py" \
        2>&1 | tee "$OUT_DIR/infer.log"
else
    swift infer "${COMMON[@]}" \
        --infer_backend sglang \
        --sglang_dp_size "${DP_SIZE:-8}" \
        --sglang_tp_size 1 \
        --sglang_mem_fraction_static "${SGLANG_MEM:-0.9}" \
        --sglang_context_length "${SGLANG_CTX:-28000}" \
        --sglang_kv_cache_dtype "${SGLANG_KV:-auto}" \
        2>&1 | tee "$OUT_DIR/infer.log"
fi

echo "--- done: $RESULT ($(wc -l < "$RESULT") rows)"
[[ "$MODE" == "head" ]] && echo "--- head logits: $OUT_DIR/cls_logits.jsonl ($(wc -l < "$OUT_DIR/cls_logits.jsonl" 2>/dev/null || echo 0) rows)"
exit 0
