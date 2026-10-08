# Training-speed optimization assessment (DALI / Liger / Flash-Attention)

Context: 30-Sep Qwen3.5-4B CT slow-fast run, 8×H100 (qrnd4), ~**50 s/it** (grad_accum 3 → ~16–17 s/micro-batch,
`per_device_train_batch_size 3`). Goal: find speedups. **Investigation only — no code changes made.** File:line
refs are from the current working tree.

## STATUS (2026-10-01)
- **MEASURED: compute-bound.** `nvidia-smi dmon -s u` on qrnd4 during training showed **sm% ~94–100% on 7/8
  GPUs**; GPU 0 sits lower (rank-0 logging/collective overhead), not data starvation. → **DALI is NOT worth it**
  (GPUs aren't waiting on data); the wins are compute-side.
- **DONE: Flash-Attention** on all three sites (§1) — implemented + CPU-tested (12/12). No config change needed
  (rides the existing `--attn_impl flash_attention_2`).
- **TODO: Liger** verify/complete (§2) — one command.
- **DROPPED: DALI** (§3) — deprioritized given compute-bound; assessment kept for reference.

## 0. FIRST, measure: data-bound or compute-bound? (decides everything)
The single most decisive number we do **not** yet have is **GPU utilization / dataloader wait**. It sorts the three
integrations:
- **High GPU util (>90%, steady)** → compute-bound → **Liger + Flash-Attention** are the wins; **DALI ≈ 0**.
- **Low/sawtooth util (GPU idling between steps)** → data-bound → **DALI / data-pipeline** is the win.
How to get it cheaply (no job restart): on qrnd4 during a step, `nvidia-smi dmon -s u -c 60` (watch the `sm%`
column), or add the HF Trainer flag that logs dataloader time, or a short `torch.profiler` window. **Do this before
investing in DALI** — the CT pipeline is heavy enough (below) that data-bound is plausible, but the 4B LLM on ~8k-token
cls_json sequences is also heavy, so it's genuinely uncertain. Everything below is ranked assuming we then confirm.

---

## 1. Flash-Attention — IMPLEMENTED (2026-10-01)

**Implemented:**
- **3D encoder** now loads with `attn_implementation` from `--attn_impl` (loader.py: new `attn_impl` param threaded
  from attach.py; `flash_attn`→`flash_attention_2` mapping; **retry-without-attn fallback** if a tower rejects it so
  the load never hard-fails). → the Qwen3.5 vision tower's spatial attention runs FA2 on the H100.
- **STA temporal attention** (sta.py): eager matmul+softmax → `F.scaled_dot_product_attention` with a band
  `attn_mask` (dispatches to FA2 / mem-efficient on CUDA, math on CPU; band always includes the diagonal so no
  all-masked row).
- **Slow-fast cross-attention** (slowfast.py): eager → SDPA with the key-padding mask; the all-masked sample (the
  distributed dummy) is handled NaN-safely (attend-all then zero the sample's output), preserving the old
  "no slow tokens → 0 contribution" semantics and finite grads.
- **Why SDPA not raw `flash_attn`:** SDPA *is* FlashAttention in PyTorch (auto-dispatches to the FA2 kernel) AND
  accepts the band / padding masks the dense `flash_attn_func` can't, with a CPU/math fallback for tests. The tower
  itself uses explicit `flash_attention_2`. (Raw varlen `flash_attn` for the two masked modules is possible but adds
  `cu_seqlens` plumbing + loses the CPU fallback — deferred unless profiling flags them.)
- **Tests:** `tests/vision_3d/test_flash_attn_sdpa.py` (6: SDPA≡eager banded/full/masked in fp64; identity-at-init;
  variable gt; fully-masked finite+zero+finite-grad) + existing `test_slowfast_crossattn.py` (6) all green.
- **Payoff:** mainly **memory** (no eager `tokens×tokens` matrix — this is what OOM'd the batch-2 AUROC run) → room
  to raise batch/`max_tokens`; plus some throughput on the encoder/cross-attn.

**Original state (verified, pre-change):**
- LLM decoder: `--attn_impl flash_attention_2` → applied to the language model (standard ms-swift path; the one
  verified contrast below shows the 3D tower is the exception). `VLLM_FORCE_FA2=1` guards the H100 FA3 PTX crash
  (see [[h100-fa3-ptx-force-fa2]]).
- **3D encoder (`model.visual_3d`) gets NO FA2.** `load_vision_3d_source_model` (loader.py:78,93) calls
  `from_pretrained` with only `trust_remote_code/torch_dtype/low_cpu_mem_usage` — **no `attn_implementation`** — and
  `attach_vision_3d_encoder` (attach.py:105-111) doesn't pass one either. So the secondary Qwen3.5 vision tower runs
  the transformers **default** backend (sdpa/eager), not FA2, on every volume forward.
- **STA temporal attention is eager manual softmax** (sta.py:101-109: `matmul(q,k)/sqrt(hd)` → `masked_fill` →
  `softmax(float())` → `matmul(attn,v)`). No SDPA/FA.
- **Slow-fast cross-attention is eager manual softmax** (slowfast.py:133-139), with a `nan_to_num` for fully-masked
  rows (padding) — relevant to any SDPA swap.

**What to add:** (a) pass `attn_implementation='flash_attention_2'` (or `'sdpa'`) when building the 3D encoder in
loader.py; (b) convert STA and the cross-attn cores to `F.scaled_dot_product_attention` (SDPA) — SDPA takes an
additive/boolean mask so the band-mask (STA) and padding-mask (cross-attn) port directly, and it auto-dispatches to
the FA/mem-efficient kernel.

**Expected gain:** the vision encoder + cross-attn are a minority of total FLOPs vs the 4B LLM, so throughput gain is
likely **single-digit–~15%**, but the bigger payoff is **memory**: eager attention materializes the full
`(tokens×tokens)` matrix (the batch-2 AUROC OOM was literally the STA eager qkv, sta.py:133-ish in the encoder). SDPA
removes that materialization → headroom to raise batch / `max_tokens`, which indirectly speeds training.

**Risks:** keep FA2, not FA3, on H100 (PTX crash). The cross-attn `nan_to_num` for all-masked rows must be preserved
(SDPA with a fully `-inf` row yields NaN) — handle by masking those query rows out or adding a tiny self-term. STA is
a custom module inserted between spatial-attn and MLP (sta.py:132-143) so it needs its own SDPA conversion regardless
of the tower backend.

---

## 2. Liger — VERIFIED (2026-10-01): already near-complete; only RoPE is off (deliberately)
**Verified** against the installed `liger_kernel` (transformers 5.9.0): `apply_liger_kernel_to_qwen3_5` **exists**, and
HF's `use_liger_kernel` routes `model_type=qwen3_5` to it. Its **defaults**:
```
rms_norm = True        # ON  -> fused
swiglu = True          # ON  -> fused
fused_linear_cross_entropy = True   # ON  (this is the skip_logits path we already handle)
cross_entropy = False  # OFF (correct -- FLCE supersedes plain CE; you don't want both)
rope = False           # OFF  <-- the only unfused LLM kernel
```
So **RMSNorm + SwiGLU + fused-linear-CE are already active** on the Qwen3.5 LLM — Liger is NOT "just CE". The only
gap is **RoPE**, and it is off **by the library's own default for qwen3_5** — almost certainly deliberate (Qwen3.5
uses an mrope / rope-scaling variant the generic Liger RoPE kernel may not reproduce exactly). **Do NOT flip
`rope=True` blindly** — if we want it, enable it explicitly *with a numerical-equivalence check* against the HF RoPE
on a forward pass first; the upside is only a few %, the downside is silent positional-encoding corruption.
- **Vision towers:** neither the native 2D tower nor the secondary 3D encoder (`model.visual_3d`, attached after HF's
  patch runs) is Liger-patched — their RMSNorm/SwiGLU are unfused. The 3D encoder could be patched manually
  (`apply_liger_kernel_to_qwen3_vl`-style RMSNorm/SwiGLU on its blocks); secondary win, optional.
- **Bottom line:** Liger is effectively complete for the LLM. No change needed unless we deliberately validate+enable
  RoPE. The earlier "maybe only CE is on" worry is resolved — it was unfounded.

### (original, pre-verification) notes
## 2b. Liger — confirm the kernel set, then complete it (medium win, needs one check)

**Current state (verified):**
- `--use_liger_kernel true` → ms-swift relies on **HF Trainer's native** `use_liger_kernel` (which calls
  transformers' `_apply_liger_kernel_to_instance`, mapping `model_type` → the liger patcher) **plus** a fork shim
  `_patch_liger_kernel` (arguments.py:240-251) that only wraps `LigerForCausalLMLoss` to drop `logits_to_keep`/
  `cu_seq_lens` kwargs and make hidden_states contiguous. The fork does **not** itself choose which kernels are swapped.
- **Fused-linear-CE is active** and gated train-only (`skip_logits = self.training`), with the fork forcing
  `skip_logits=True` at eval to avoid the full-logits OOM (seq2seq_trainer.py:470-475; see [[slowfast-liger-gradckpt-gotchas]]).

**THE open question (one command to answer):** does the installed `liger_kernel` recognize the **`qwen3_5`**
`model_type` and swap **RMSNorm + RoPE + SwiGLU-MLP**, or only the CE loss? Qwen3.5 is a new arch; if liger_kernel has
no `apply_liger_kernel_to_qwen3_5`, those three fusions are **silently not applied** and only fused-CE is active.
Verify (warm FS):
```
python -c "from liger_kernel.transformers import monkey_patch as m; print([k for k in dir(m) if 'qwen3' in k.lower()])"
# and inspect transformers' MODEL_TYPE_TO_APPLY_LIGER_KERNEL_FN for 'qwen3_5'
```
- If RMSNorm/RoPE/SwiGLU are **missing** → "complete incorporation" = register a `qwen3_5` patcher (reuse the
  qwen3 / qwen2 kernels — same RMSNorm/RoPE/SwiGLU math) so those three get fused. **Expected ~10–20% LLM throughput +
  memory.**
- If already applied → little to gain on the LLM.
- Either way, the **vision towers are not Liger-patched** (native 2D + 3D encoder RMSNorm/MLP run unfused). The 3D
  encoder's RMSNorm/SwiGLU are a smaller share, but fusing them is a secondary win once the LLM set is confirmed.

**Risks:** low — kernel-level, numerically equivalent. The known gotchas are already handled (eval skip_logits, the
loss shim). Main caveat: a wrong `model_type` mapping could silently no-op (hence the verify step).

---

## 3. DALI — DROPPED (measurement says compute-bound; it can't help)
**Decision (2026-10-01):** the `nvidia-smi dmon` measurement showed GPUs at ~94–100% sm% — **compute-bound**, not
data-bound. DALI accelerates the data pipeline, which is **not** the bottleneck, so it would not raise throughput;
building it would be wasted effort. Not implemented. The pipeline assessment below is retained for reference in case
the balance shifts (e.g. after the compute-side wins + a smaller model make the GPUs faster and the data pipeline
relatively heavier — re-measure then). If it ever does become data-bound, the cheap wins (cache windowed volumes,
more workers, GPU-side windowing) come before a full DALI port.

### (reference) why the pipeline *would* be DALI's target if we were data-bound

**Current state (verified):** the entire volume pipeline is **CPU**, in dataloader workers:
- s3 → local blob via `rnd_data_lake.cache.get_cached_path` (ct_volume_io.py:32-48); content-addressed cache, **no
  re-download across epochs**; fetch is lazy/synchronous per worker, no prefetch in-repo.
- decode: `safetensors.torch.load_file` (CPU) → `read_hu_volume` (ct_volume_io.py:76-116).
- **trilinear `F.interpolate` resize** on CPU (ct_volume_io.py:119-123; smart-resize 126-162).
- **11-channel HU windowing** = Python list-comp of ~11 full-volume `clamp`/scale passes + stack (ct_windowing.py:82-89,
  33-53) → **11× memory blow-up**.
- per-channel standardize, permute+contiguous, (optional) augment — all CPU float32.
- Payload: `(T,C,H,W)` ≈ `(64,11,224,224)` fixed-mode → **~141 MB float32 / ~70 MB bf16 per volume**, built per step
  in workers and shipped H2D. No GPU preprocessing anywhere; first GPU touch is the `visual_3d` forward in
  `_post_encode`.

**The two CPU hotspots** (trilinear resize, 11× windowing) map cleanly to GPU kernels — exactly DALI's domain.

**Expected gain:** **large IF data-bound, ~0 IF compute-bound.** DALI would move resize/window/normalize/cast to the
GPU and overlap with compute. Caveats specific to us:
- DALI natively decodes JPEG/video, **not** medical safetensors/NIfTI/DICOM HU volumes — so the CPU *read* + the s3
  `rnd_data_lake` fetch stay CPU; DALI mainly accelerates the **tensor ops after the read** (resize/window/normalize).
- That's still the bulk of the CPU cost, but it means a full DALI external-source pipeline, not a drop-in reader.

**Cheaper wins to try BEFORE DALI (if data-bound):**
1. **Pre-materialize windowed volumes**: cache the `(T,C,H,W)` windowed/resized tensor (bf16) to the local cache once,
   so epoch-2+ skips decode+resize+windowing entirely. Biggest bang, least code — the 11× windowing + trilinear are
   done once, not every epoch.
2. **Raise `dataloader_num_workers` / `prefetch_factor`** (confirm current values in the `.sh`) — more CPU parallelism
   hides decode behind compute; near-free.
3. **Drop channels we don't need**: C=11 is a 11× cost; if the encoder patch-embed uses fewer effective channels,
   trimming `--ct_windows` cuts decode + H2D proportionally.
4. **GPU-side windowing without full DALI**: do the windowing/normalize as a torch op on GPU right before the encoder
   (the HU volume ships as 1 channel, windows expand on-device) — kills the 11× H2D blow-up and the windowing loop with
   a few lines, no DALI dependency. Often captures most of DALI's benefit at a fraction of the effort.

**Risk:** DALI is a heavy new dependency + a rewrite of the worker-side pipeline; only worth it if (1) data-bound is
confirmed AND (2) the cheaper wins above don't close the gap.

---

## Recommended order (updated)
1. ~~Measure GPU util~~ — **DONE: compute-bound** (sm ~94–100%).
2. ~~Flash-Attention~~ — **DONE** (§1): 3D-encoder FA2 + SDPA for STA & cross-attn; tested.
3. **Verify + complete Liger** (§2) — NEXT: one command to see if RMSNorm/RoPE/SwiGLU are applied for `qwen3_5`;
   register the patcher if missing. Medium win, low risk, compute-side (so it directly helps the compute-bound run).
4. **Smaller model** — the Qwen3.5-**2B** config roughly halves per-step compute vs 4B; the single biggest lever for
   the "25 days" problem. Config prepared (see `examples/train/liger/*_2b_*`).
5. **DALI — DROPPED** (§3): compute-bound, so it can't help. Revisit only if a future re-measure shows data-bound.
6. **Other compute levers to consider** (not yet done): with FA2 freeing memory, test disabling gradient
   checkpointing on some layers (recompute is ~20–30% of step time); and confirm sequence lengths aren't inflated
   (the cls_json roll-call is ~8k tokens — the dominant per-step compute).

Links: [[30sep-slowfast-run]], [[h100-fa3-ptx-force-fa2]], [[slowfast-liger-gradckpt-gotchas]], [[slowfast-liger-gradckpt-gotchas]].
