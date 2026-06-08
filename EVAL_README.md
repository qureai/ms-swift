# Qwen3.5-VL Inference with SGLang + ms-swift

Environment setup for running multimodal Qwen models via [ms-swift](https://github.com/modelscope/ms-swift) with an [SGLang](https://github.com/sgl-project/sglang) backend. The stack pins compatible versions of SGLang, FlashInfer, Transformers, flash-attn, and supporting CUDA wheels, then runs streaming inference across 8 GPUs.

## Requirements

- Linux with NVIDIA GPUs (tested config below assumes 8 visible GPUs)
- CUDA 12.6 / 12.8 compatible drivers
- Python with [`uv`](https://github.com/astral-sh/uv) installed
- `pip` available in the same environment

## Installation

Run the steps in order. Each command pins `torch` to whatever version was installed in the previous step so the build chain stays consistent.

### 1. Core stack — SGLang, Triton, Torch, FlashInfer, TorchAO

```bash
uv pip install --force-reinstall \
    "sglang[all]==0.5.10" triton torch \
    https://github.com/flashinfer-ai/flashinfer/releases/download/v0.6.7.post2/flashinfer_python-0.6.7.post2-py3-none-any.whl \
    https://download.pytorch.org/whl/cu126/torchao-0.9.0-py3-none-any.whl \
    https://www.piwheels.org/simple/packaging/packaging-24.2-py3-none-any.whl \
    --no-cache-dir --no-build-isolation --prerelease=allow \
    --find-links https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.7.post2 \
    --extra-index-url https://flashinfer.ai/whl/cu128 \
    --extra-index-url https://sgl-project.github.io/whl/cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128
```

### 2. ms-swift (training/inference frontend)

```bash
pip install -U ms-swift "torch==$(python -c 'import torch; print(torch.__version__)')"
```

### 3. Transformers + Qwen-VL utilities + PEFT + Liger

```bash
uv pip install -U \
    "transformers==5.2.*" \
    "qwen_vl_utils>=0.0.14" \
    peft liger-kernel \
    "torch==$(python -c 'import torch; print(torch.__version__)')"
```

### 4. Flash Linear Attention

```bash
uv pip install -U "flash-linear-attention>=0.4.2" \
    --no-build-isolation \
    "torch==$(python -c 'import torch; print(torch.__version__)')"
```

### 5. Causal Conv1D (from source)

```bash
uv pip install -U git+https://github.com/Dao-AILab/causal-conv1d \
    --no-build-isolation \
    "torch==$(python -c 'import torch; print(torch.__version__)')"
```

### 6. Flash Attention 2

```bash
uv pip install "flash-attn==2.8.3" --no-build-isolation
```

### 7. DeepSpeed

```bash
uv pip install deepspeed "torch==$(python -c 'import torch; print(torch.__version__)')"
```

### 8. Mistral Common (tokenizer utilities)

```bash
pip install -U mistral_common
```

## Inference

Streaming inference with Qwen3.5-4B across 8 GPUs using the SGLang backend. Vision/video token limits are tuned for ~1K image tokens and short video clips (16 frames max).

```bash
IMAGE_MAX_TOKEN_NUM=1024 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift infer \
    --model Qwen/Qwen3.5-4B \
    --enable_thinking false \
    --stream true \
    --use_hf true \
    --infer_backend sglang \
    --sglang_tp_size 8 \
    --temperature 1 \
    --top_p 0.5 \
    --top_k 50 \
    --max_new_tokens 2048 \
    --repetition_penalty 1.0 \
    --num_beams 1 \
    --stop_words '<|im_end|>' \
    --logprobs false \
    --top_logprobs 0 \
    --max_batch_size 1 \
    --result_path infer_results.jsonl \
    --write_batch_size 1 \
    --sglang_context_length 32768 \
    --sglang_mem_fraction_static 0.9
```

### Key flags

| Flag | Purpose |
|---|---|
| `IMAGE_MAX_TOKEN_NUM` | Max visual tokens per image |
| `VIDEO_MAX_TOKEN_NUM` | Max visual tokens per video frame |
| `FPS_MAX_FRAMES` | Frame cap when sampling videos |
| `--sglang_tp_size` | Tensor-parallel degree (match GPU count) |
| `--sglang_context_length` | KV-cache context window (32K here) |
| `--sglang_mem_fraction_static` | Fraction of GPU memory reserved for static weights/KV |
| `--enable_thinking false` | Disables Qwen's `<think>` reasoning traces |
| `--use_hf true` | Pull model from Hugging Face Hub |
| `--result_path` | JSONL output for batched runs |

## Notes

- The `--no-build-isolation` flag is required for `flash-attn`, `causal-conv1d`, and `flash-linear-attention` so they compile against the already-installed `torch` rather than fetching a fresh one.
- The repeated `"torch==$(python -c 'import torch; print(torch.__version__)')"` pattern locks each follow-up install to the exact torch version pulled in step 1 — avoid skipping this or pip may silently upgrade torch and break the FlashInfer/flash-attn ABI.
- `--force-reinstall` in step 1 is intentional: it ensures the CUDA-matched wheels from the FlashInfer / SGLang / PyTorch indexes are used instead of any cached CPU-only or mismatched wheels.
- Multiple `--extra-index-url` entries are needed because the stack pulls cu128-built wheels from three different hosts (FlashInfer, SGLang, PyTorch).

