#!/usr/bin/env bash
# Burned-in PHI audit over the 25,487-study "easy" eval subset using stock Qwen/Qwen3.5-27B
# (no fine-tune) served through sglang, 8-way data-parallel on 8x H200.
#
# Input  : $OUT/infer_input.jsonl        (build_phi_infer_input.py)
# Output : $OUT/result.jsonl             (one row per image, response = strict JSON)
#
# Usage: run_phi_audit_25k.sh [pilot]     # `pilot` runs the 200-image subdir instead
set -euo pipefail

CONDA_ENV=/home/ubuntu/miniconda3/envs/ms-swift-may
export PATH="$CONDA_ENV/bin:$PATH"          # sglang JIT kernels need `ninja` on PATH
export HF_HOME=/fastdata4/huggingface
export HF_HUB_CACHE=/fastdata4/huggingface/hf_hub
# Burned-in PHI is small text in the image corners, so give the vision tower a much
# larger token budget than the 1.8k default -- ~4096 visual tokens ~= a 2048x2048 grid,
# which keeps ~20px corner text legible after downscaling from the native ~3000px.
export IMAGE_MAX_TOKEN_NUM=4096
# `python swift/cli/infer.py` puts swift/cli on sys.path[0], so `import swift` would resolve
# to the site-packages copy -- which lacks this tree's sglang_engine.py fix remapping
# images->image_data for sglang>=0.5 (and its load_image guard). Force the working tree.
export PYTHONPATH=/fastdata4/sahil_work/VLM_qure/vlm/swift:${PYTHONPATH:-}
# Inputs are pre-screened, so this should never fire; if an image is somehow undecodable we
# want a loud failure rather than the silent random-noise placeholder, which would be
# scored as a confident "no patient info" on an image the model never saw.
export SKIP_BROKEN_IMAGES=1

# NOTE: run on the *screened* input (screen_phi_images.py). swift's inference path either
# crashes on an undecodable image or silently swaps in a random-noise placeholder; the
# latter would produce a confident "no patient info" for an image nobody actually read.
BASE=/fastdata4/sahil_work/VLM_qure/eval_outputs/Qwen3.5-27B_phi_audit_25k
OUT="$BASE/screened/retry"


cd /fastdata4/sahil_work/VLM_qure/vlm/swift

python swift/cli/infer.py \
    --model Qwen/Qwen3.5-27B \
    --model_type qwen3_5 \
    --template qwen3_5 \
    --torch_dtype bfloat16 \
    --infer_backend sglang \
    --sglang_dp_size 8 \
    --sglang_tp_size 1 \
    --sglang_mem_fraction_static 0.85 \
    --sglang_context_length 16384 \
    --sglang_disable_custom_all_reduce true \
    --val_dataset "$OUT/infer_input.jsonl" \
    --result_path "$OUT/result.jsonl" \
    --write_batch_size 500 \
    --max_new_tokens 512 \
    --max_length 16384 \
    --temperature 0.5 \
    --top_p 0.9 \
    --top_k 20 \
    --repetition_penalty 1.0 \
    --enable_thinking false \
    --use_hf true \
    --val_dataset_shuffle false \
    2>&1 | tee "$OUT/infer.log"
