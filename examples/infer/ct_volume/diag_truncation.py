"""One-shot GPU diagnostic for the CT-volume front-truncation bug.

Reproduces a single inference through the SAME TransformersEngine path as run_ct_infer, and
instruments `template.get_generate_ids` to print the raw generated sequence length, the
`num_prompt_tokens` used to slice it, and the FULL decoded raw sequence (so we can see exactly
where the prompt ends and how many response tokens are being chopped).
"""
import os, sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift")

from swift.arguments import InferArguments
from swift.pipelines.utils import prepare_model_template
from swift.infer_engine import TransformersEngine, InferRequest, RequestConfig

CKPT = "/cache/fast_data_nas71/janhavi/VLM_qure/vlm/swift/vlm_ckpts/1308_qwen3_5_4b_atlas_3d_ct/v1-20260814-115348/checkpoint-26500"
VOL = "s3://rnd-data-lake/safetensors/1.3.6.1.4.1.55648.0001390543684078601688995807461081.2.safetensors"
PROMPT = ("Generate a structured report in JSON for the given CT volume with the following fields: "
          "findings, summary, complete_reasoning, impression, recommendations, and prominence_score.")

args = InferArguments(model=CKPT, max_new_tokens=96, attn_impl="sdpa", torch_dtype="bfloat16")
model, template = prepare_model_template(args)
engine = TransformersEngine(model, template=template, max_batch_size=1)
tok = template.tokenizer

# tokens of the expected opener the model should emit first (to compare against what's dropped)
opener = '```json\n{\n  "findings": "'
print(f"[DIAG] opener tokenizes to {len(tok.encode(opener, add_special_tokens=False))} tokens: {opener!r}", flush=True)

_orig = template.get_generate_ids
def _wrapped(generate_ids, num_prompt_tokens):
    try:
        gi = generate_ids
        shape = tuple(gi.shape) if hasattr(gi, "shape") else ("list", len(gi))
        print(f"[DIAG] raw generate_ids shape={shape}  num_prompt_tokens={num_prompt_tokens}", flush=True)
        row = gi[0] if (hasattr(gi, "ndim") and gi.ndim == 2) else gi
        ids = row.tolist() if hasattr(row, "tolist") else list(row)
        full = tok.decode(ids)
        print(f"[DIAG] raw len(ids)={len(ids)}", flush=True)
        print(f"[DIAG] raw decoded FIRST 220 chars: {full[:220]!r}", flush=True)
        print(f"[DIAG] raw decoded LAST 120 chars:  {full[-120:]!r}", flush=True)
        # what do the first num_prompt_tokens vs the slice look like?
        if hasattr(gi, "ndim"):
            print(f"[DIAG] decoded[num_prompt_tokens:] FIRST 120: "
                  f"{tok.decode(ids[num_prompt_tokens:])[:120]!r}", flush=True)
    except Exception as e:
        print("[DIAG] wrap error:", e, flush=True)
    return _orig(generate_ids, num_prompt_tokens)
template.get_generate_ids = _wrapped

req = InferRequest(
    messages=[{"role": "system", "content": "You are a helpful medical assistant."},
              {"role": "user", "content": f"{PROMPT} <video>"}],
    videos=[VOL])
resp = engine.infer([req], RequestConfig(max_tokens=96, temperature=0.0))
print("[DIAG] FINAL response FIRST 220 chars:", repr(resp[0].choices[0].message.content[:220]), flush=True)
print("[DIAG] done", flush=True)
