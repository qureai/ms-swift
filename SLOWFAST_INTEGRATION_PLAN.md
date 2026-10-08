# Slow-Fast Video-MLLM Integration Plan (CT 3D VLM)

Design/implementation plan for adding a **slow-fast** visual-token path to the swift CT-VLM fork
(Qwen3.5-4B LLM + Qwen3.5 vision tower as a 3D encoder). **No code has been written yet** — this doc is the
spec and the test plan. Reference: SHI-Labs *Slow-Fast Video Multimodal LLM* (`hybrid_decoder_layer.py`,
`llava_qwen2_slowfast.py`, `llava_arch.py`), adapted to CT volumes and our existing modules.

---

## 0. Goal & guiding principles

**What it buys us:** the LLM self-attention + MLP run over a *short* sequence (a small set of **fast** tokens),
while full spatial/temporal detail stays available as **slow** tokens that a few decoder layers cross-attend to
(read-only K/V, never MLP'd). Shorter sequences -> faster LLM forward/backward, eval fits on smaller GPUs,
grounding detail preserved. It does **not** reduce the vision-encoder cost (slow tokens still come from the full
encoder pass).

**Principles (non-negotiable):**
1. **Mirror the STA pattern** (`swift/model/vision_3d/sta.py`): a gated, **zero-initialised** attention block
   inserted into selected layers, recorded in the saved config, and **re-installed on reload before weights load**.
2. **Backwards compatible by default:** master switch `--vision_3d_slowfast` defaults to `false`. When off, every
   new code path is skipped and behaviour / state-dict / `num_volume_tokens` are byte-identical to today.
3. **Everything with learnable weights is a registered `nn.Module`** so full-FT saves it automatically.

**Key dims (Qwen3.5-4B):** `llm_hidden = 2560`; encoder output `enc_dim = 2560`. Read `num_attention_heads`,
`num_key_value_heads`, `head_dim = hidden/num_heads`, `num_hidden_layers` from `config.get_text_config()` at
build — **never hardcode**.

---

## 1. New module — cross-attention hybrid layer

**New file:** `swift/model/vision_3d/slowfast.py` (structured like `sta.py`).

### 1.1 `LLMCrossAttnBlock(nn.Module)` — text queries the slow tokens

| submodule | shape | notes |
|---|---|---|
| `norm` | RMSNorm(`llm_hidden`) | pre-norm on text hidden before querying |
| `q_proj` | `Linear(llm_hidden -> n_heads*head_dim)` | query from text |
| `kv_proj` | `Linear(slow_dim -> n_kv_heads*head_dim*2)` | K,V from slow tokens (GQA) |
| `o_proj` | `Linear(n_heads*head_dim -> llm_hidden)` | output back to hidden |
| `gate` | `nn.Parameter(zeros(1))` (+ optional `Linear(llm_hidden,1)+tanh`) | **zero-init** => 0 at init |

`forward(hidden, slow_kv, slow_mask)`:
```
q = q_proj(norm(hidden))                       # (B, T_txt, n_heads, head_dim)
k, v = kv_proj(slow_kv).chunk(2)               # (B, T_slow, n_kv_heads, head_dim) -> repeat_kv for GQA
ctx  = softmax(q k^T / sqrt(head_dim) + pad_mask(slow_mask)) v   # text->slow, NO causal mask
return hidden + gate * o_proj(ctx)             # gated residual (0 at init)
```
- **No causal mask** (text may attend to all slow tokens); `slow_mask` masks padding only.
- **Open decision (Sec.11):** RoPE on `q`. Default proposal: **no RoPE** (slow tokens have no text position).

### 1.2 `SlowFastDecoderLayer(nn.Module)`
Wraps one original Qwen3.5 decoder layer (reuses its `self_attn`, `mlp`, norms), inserts the cross-attn block
**after self-attn, before MLP** (identical placement to `STAVisionBlock`):
```
h = orig_self_attn_sublayer(h, ...)      # unchanged
h = crossattn_block(h, slow_kv, slow_mask)    # NEW gated residual (0 at init)
h = orig_mlp_sublayer(h, ...)            # unchanged
```
Must forward with the original layer's exact signature (attention_mask, position_ids, past_key_value,
position_embeddings, cache flags) so it is a drop-in for the HF decoder layer.

### 1.3 `install_slowfast(llm, layer_idxs, slow_dim, cfg) -> meta`
- In-place replace `layers[i]` with `SlowFastDecoderLayer` for `i in layer_idxs`; inherit each layer's
  `gradient_checkpointing` flag (as `install_sta` does).
- Register a **decoder forward-pre-hook** that stashes the batch slow tensor + mask onto each hybrid layer before
  forward (the `_sta_grid_thw` stash idiom).
- Return `meta = {slowfast_enabled, layer_idxs, slow_dim, n_heads, n_kv_heads, head_dim, gate_type}` for the config.

---

## 2. Fast / slow token production

**File:** `swift/model/vision_3d/backbone.py`, `Vision3DBackbone.forward` (variable-token path).
After the encoder emits per-volume tokens `(M_i, enc_dim)` with grid `(gt, gh, gw)`:

### 2.1 Fast branch — 3D adaptive **avg + max** pool (reduces over z, h AND w)
- Reshape tokens to `(gt, gh, gw, enc_dim)` using `volume_grid_thw`.
- **Adaptive 3D pool to a fixed target grid** `(Tf, Hf, Wf)` (arg `--vision_3d_fast_grid`) -> **bounded**
  `fast_count = Tf*Hf*Wf` regardless of volume size. The pool window is **3D and spans the z (slice) axis** as
  well as h,w — CT slices are not redundant, so z is **pooled, not subsampled**.
- **avg + max, reduced over the FULL 3D window (including z):** for each `(Tf,Hf,Wf)` window compute
  `avg = mean over the z*h*w elements` and `max = max over the same z*h*w elements`, per channel, then
  **concatenate along the feature dim** -> `(fast_count, 2*enc_dim)`.
  - The **max over z** is what carries a single-slice / small finding's peak up into the fast overview (avg alone
    would dilute it across the pooled slices).
- Project: `proj_fast = Linear(2*enc_dim -> llm_hidden)` when `fast_pool=avgmax` (else `enc_dim -> llm_hidden`).
  Output `(fast_count, llm_hidden)`. These are spliced into `<video>`. Token count stays `fast_count`.

### 2.2 Slow branch — full detail for cross-attn K/V
- Keep the full encoder tokens `(M_i, enc_dim)` (optionally cap via 3D adaptive avg-pool to
  `--vision_3d_slow_max_tokens`). `slow_dim = enc_dim` (the block's `kv_proj` maps `slow_dim -> kv`).
- Stash per-volume slow tokens on the backbone (`self._slow_tokens`) — same idiom as `self._aux_feats`.

`num_tokens()` / `num_volume_tokens()` must return **`fast_count`** when slow-fast is on.

---

## 3. Splice + conditioning

**File:** `swift/template/templates/volume_mixin.py`.
- `num_volume_tokens()` -> `fast_count` when slow-fast on (drives `<video>` expansion in `_encode`/`_extend_tokens`
  to the short fast length; the sequence-length win originates here).
- `_post_encode()`:
  1. `_splice_volume_embeds(...)` splices the **fast** tokens into `<video>` positions (unchanged path, fewer tokens).
  2. **New:** pad `backbone._slow_tokens` to `(B, T_slow_max, enc_dim)` + build `slow_mask (B, T_slow_max)`, push
     onto the hybrid layers (install_slowfast conditioning) before the LLM forward.
- Collator / `_prepare_volumes`: carry per-volume slow-token counts for padding+mask (mirrors variable `token_lens`).

---

## 4. Attach + optimizer wiring

**File:** `swift/model/vision_3d/attach.py`. When `args.vision_3d_slowfast`:
- Build `proj_fast` (and any slow projection) inside `visual_3d`.
- `layer_idxs = range(crossattn_min_layer, crossattn_max_layer, crossattn_every_n)`.
- `install_slowfast(<language_model>, layer_idxs, slow_dim=enc_dim, cfg=...)`.
- **Register the new params in the optimizer group** (extend the `visual_3d`/`vision_tower` registration so
  `--vit_lr` or `--vision_3d_crossattn_lr` targets them — same fix that stopped `visual_3d` being dropped from
  the vit_lr group).
- **Record slow-fast config** into the saved `vision_3d_config` (enabled, layer_idxs, dims, fast_grid, fast_pool,
  slow_max_tokens).
- **ZeRO ordering:** install runs before DeepSpeed/optimizer init (attach.py already runs pre-trainer) so new params
  are sharded/tracked.

---

## 4A. Representation health — projector normalization + anti-collapse (from spectrum analysis)

**Motivation (encoder spectrum, checkpoint-1000, 24 vision blocks):**
- Final block (23) is **near rank-1** (effective rank 1.01) with feature **norm ~3524** — a giant *shared*
  baseline direction almost identical across tokens. The projector eats this block **raw, un-normalised**
  (`backbone.py:133`), so a near-constant huge-norm vector dominates the `Linear` input.
- Mid blocks (10-19) sit at effective rank **~2** — and this is measured on **centred** tokens
  (`o - o.mean(0)`, `encoder_spectrum.py:76`), i.e. *after* the shared baseline is removed. So the
  token-to-token variation is **intrinsically ~2-dimensional**, not merely baseline-dominated.
- POST-proj reldiff fell **0.33 (ckpt-500) -> 0.16 (ckpt-1000)** — input sensitivity is being *eroded*
  during fine-tuning at `vit_lr 3e-4` (collapse in progress toward the ~0.004 floor).

Two **distinct** problems, two **distinct** fixes — do not conflate them:

### (a) Pre-projector LayerNorm — fixes THROUGHPUT, **not** rank
- **What it does:** `nn.LayerNorm(enc_dim)` at the front of `self.proj` (and `proj_fast` / `proj_slow` when
  slow-fast is on) strips the ~3524 shared-baseline direction so the small input-dependent variation actually
  survives the `Linear` and reaches the LLM.
- **What it does NOT do:** it will **not raise effective rank.** LayerNorm is a *per-token* op (subtract each
  token's scalar mean, divide by its scalar std, shared gamma/beta) — it rescales rows and reweights columns
  but adds no new dimensions to the centred variation subspace. Rank-2 in -> ~rank-2 out. Because the spectrum's
  rank is already computed on centred tokens, LN barely moves that number. Keep this labelled honestly.
- **Backward compat:** gated by `--vision_3d_proj_prenorm` (default `false` -> the layer is `nn.Identity`, so
  existing checkpoints load byte-identical and behaviour is unchanged). When enabled on an *old* checkpoint,
  the LN is new-init (affine gamma=1/beta=0) and loads with `strict=False`.

### (b) Variance + covariance anti-collapse loss — the actual RANK fix
> **Diagnostic-confirmed:** the collapse is a *correlation* collapse (channels vary but are linearly
> correlated -> ~2 effective directions). The **covariance/decorrelation term is the operative one**;
> the variance floor is secondary insurance. Weight `cov_w` accordingly.

- **What it does:** VICReg-style auxiliary loss on the **pre-projector** tokens (already exposed via
  `_aux_feats`; for slow-fast, apply on `_slow_tokens` — the full-detail branch, since the fast branch is
  pooled and therefore intrinsically lower-rank). Per batch of tokens `Z (N, enc_dim)`:
  - **variance term:** `mean_j relu(gamma - std(Z[:,j]))` with `gamma=1` — hinges each feature dim's std up,
    preventing dimensions from collapsing to zero.
  - **covariance term:** `sum_{i!=j} Cov(Z)_{ij}^2 / enc_dim` — decorrelates dimensions, spreading energy
    across the spectrum. Together these **directly raise effective rank** (that is what VICReg optimises).
  - `L_rankreg = var_w * var_term + cov_w * cov_term`, added to the LM loss.
- **Backward compat:** gated by `--vision_3d_rankreg_lambda` (default `0.0` -> not computed, no graph nodes,
  zero overhead). No new parameters -> no checkpoint-shape change.

- **Priority: HIGH — this is the endorsed rank fix** (Phase-0 diagnostic decides whether it is *needed*, but
  it is the fix to build if collapse is confirmed).

**Implementation — code changes required:**
- **`swift/trainers/arguments.py`** (Seq2SeqTrainingArguments): add
  `vision_3d_rankreg_lambda: float = 0.0`, `vision_3d_rankreg_var_w: float = 1.0`,
  `vision_3d_rankreg_cov_w: float = 1.0`, `vision_3d_rankreg_gamma: float = 1.0` (std target),
  `vision_3d_rankreg_max_tokens: int = 2048` (token subsample cap for the covariance term — see cost note).
- **`swift/model/vision_3d/backbone.py`** (`Vision3DBackbone.forward`): when rankreg is on and `self.training`,
  stash the **pre-projector** tokens for the trainer. Reuse the existing `_aux_feats` mechanism — generalise it
  to a `self._rankreg_feats` (the concatenated `(sum_i M_i, enc_dim)` pre-proj tensor, kept in-graph, i.e. NOT
  detached). ~5 lines; no new params, no shape change. For slow-fast, stash `self._slow_tokens` instead (fast
  tokens are pooled -> intrinsically low rank).
- **`swift/trainers/seq2seq_trainer.py`** (`compute_loss`, alongside the existing presence-weight / channel
  logging): if `vision_3d_rankreg_lambda > 0` and training, pull `model.visual_3d._rankreg_feats` -> `Z (N, D)`,
  compute in **fp32**:
  - `std = sqrt(var(Z, dim=0) + 1e-4)`; `var_term = mean_j relu(gamma - std_j)`.
  - subsample `N -> min(N, rankreg_max_tokens)` rows, center -> `Zc`; `C = (Zc.T @ Zc)/(n-1)` (D x D);
    `cov_term = (C.pow(2).sum() - C.diag().pow(2).sum()) / D`.
  - `L = lambda * (var_w * var_term + cov_w * cov_term)`; add to LM loss; log `loss/rankreg`, `loss/rr_var`,
    `loss/rr_cov` to wandb. ~35 lines.
- **`examples/train/liger/*.sh`**: set `--vision_3d_rankreg_lambda`, weights, `--vision_3d_rankreg_max_tokens`.
- **Cost note:** the variance term is trivial (per-column std). The covariance term is `O(n*D^2)`; at the full
  patch-token count (`n` up to ~4096, `D`=2560) that is tens of TFLOP/step. **Subsample `n` to `rankreg_max_tokens`
  (~1-2k)** -> a few TFLOP, negligible vs the LLM. `C` itself is `D x D` = 2560x2560 (~26 MB fp32) — fine.
- **Backward compat / checkpoint:** no new parameters, so the student `state_dict` shape is unchanged; old
  checkpoints load byte-identical; `lambda=0` computes nothing (no graph nodes, zero overhead).

- **Axis (important):** VICReg's original variance/covariance are over the BATCH; our failure is **token
  uniformity**, so compute both over the **TOKEN axis** — `Z` = all patch tokens of the batch's volumes stacked
  `(N_tokens, D)`, `Var`/`Cov` reduced over `dim=0` (tokens, within + across volumes). Exact variance hinge:
  `v(Z) = (1/D) * sum_j relu(gamma - sqrt(Var(Z[:,j]) + eps))`, `gamma=1` — zero cost until a dim's std drops
  below 1, then it floors it; a constant (collapsed) representation is maximally penalised.
- **Fallback caveat (arXiv:2411.00392):** preventing dimensional collapse in the *representations* does not
  guarantee it in the encoder's *weight matrices*. If post-proj eff_rank looks healthy but grounding still
  fails, add an **orthogonality regulariser on the projector / temporal-attn weight matrices** (`||W W^T - I||_F`)
  as a second-line term. Not needed unless the representation-level floor proves insufficient.

### (c) Feature-layer selection — tap a less-collapsed block
- **What it does:** `--vision_3d_feature_layer k` feeds `hidden_states[k]` to the projector instead of the last
  block (which is the rank-1 / norm-3524 one). Earlier/mid blocks are less collapsed; a small win on its own,
  larger combined with (a)+(b). Multi-layer concat (deepstack `{12,18,23}`) is the richer variant.
- **Backward compat:** default `-1` (last block) -> current behaviour exactly. Requires the encoder to run with
  `output_hidden_states=True` (blocks are already individually addressable — the spectrum hooks proved this).

**Interaction with slow-fast:** the fast-branch avg+max pool (SS 2.1) *reduces* rank by construction (pooling
averages tokens). So route the rank-reg (b) onto the **slow** tokens, keep (a) LN on **all** projections
(`proj_fast`, `proj_slow`, base `proj`), and prefer (c) an intermediate feature layer for the slow K/V.

**Test cases (add to SS 8):**
- **8.13 proj_prenorm identity when OFF:** with `--vision_3d_proj_prenorm false`, forward output is bit-identical
  to pre-change; the prenorm module is `nn.Identity`.
- **8.14 proj_prenorm removes baseline:** feed tokens = (large constant vector + small noise); assert the
  post-LN feature norm is O(1) and the pairwise token variance is preserved (baseline gone, signal kept).
- **8.15 rankreg raises rank:** on a synthetic rank-2 token batch, one optimiser step with
  `--vision_3d_rankreg_lambda>0` increases measured effective rank; with lambda=0 the loss/graph is untouched.
- **8.16 rankreg no-op when OFF:** `lambda=0` -> loss value and `.backward()` graph identical to baseline
  (no extra nodes, no extra params in `state_dict`).
- **8.17 feature_layer selection:** `--vision_3d_feature_layer 18` makes the projector consume block-18 output
  (assert shape + that it differs from the last-block path); `-1` reproduces current output exactly.
- **8.18 checkpoint round-trip:** save with prenorm ON, reload -> LN weights restored; load an OLD checkpoint
  (no LN) with prenorm ON -> `strict=False`, LN new-init, no error.

### (d) Gram anchoring — preserve dense/patch structure during full unfreeze (DINOv3)
> **Priority: LOW — do NOT implement the Gram code yet.** Documented here for later. Prefer (b) VICReg
> as the rank fix first; revisit Gram anchoring only if VICReg + the training-recipe fixes are
> insufficient. No `--vision_3d_gram_*` code should be written until this note is lifted.

- **Rationale:** DINOv3 documents that extended full-FT improves *global* metrics while *dense* prediction
  degrades (patch features drift, lose spatial fidelity) even though global embeddings stay strong. This is
  **exactly our symptom** — POST-proj reldiff 0.33 -> 0.16 while the LM loss keeps falling. The fix anchors the
  student's patch-similarity structure to a teacher taken while dense features were still sharp.
- **Loss:** on pre-proj patch tokens, L2-normalise features `X_n (M,d)`, form Gram `G = X_n @ X_n^T (M,M)`, then
  `L_Gram = ||G_s - G_g||_F^2 / M^2`, with `G_g` from a **frozen teacher** on the *same* volume. Added to LM loss
  with weight `gram_lambda`.
- **Teacher:** a frozen second copy of the encoder, built via the existing `load_vision_3d_source_model` +
  `extract_vision_tower` + `install_sta` path pointed at `--vision_3d_gram_teacher <ckpt>`; `requires_grad_(False)`,
  `.eval()`. **ZeRO-2 (our config) shards optimiser/grads only, NOT params** -> the teacher is held plain and
  replicated per GPU, no partition/gather. Not saved into the student checkpoint (external, frozen).
- **Overhead:** +1 frozen encoder forward per step (no_grad, no backward) ~= **+5-15% wall-clock** (LLM fwd+bwd
  dominates); student Gram reuses tokens already in the graph (no 2nd student forward). Gram matmul `O(M^2 d)`
  (~2.7 GFLOP/vol at M=1024). Memory: ~1.5-2 GB bf16 teacher replicated + `M^2` Gram (4 MB @1024, 64 MB @4096;
  cap via `--vision_3d_gram_max_tokens`).
- **Preserve vs. increase (important):** Gram anchoring **preserves** the teacher's structure — it stops further
  collapse but **will not exceed the teacher's rank.** So teacher choice is the key decision:
  pretrained Qwen encoder = highest dense rank but off-domain CT relations; **early-but-adapted checkpoint
  (~ckpt-200-500)** = domain-adapted and pre-collapse (DINOv3's own recipe). **Measure rank/reldiff across early
  checkpoints and pick the sharpest domain-adapted one.** To push *above* the teacher's rank, compose Gram
  (preserve) with a small `rankreg` / VICReg term (push) from SS 4A(b).
- **Backward compat:** gated by `--vision_3d_gram_lambda` (default `0.0` -> teacher not built, no forward, no
  overhead). No new trainable params, no student-checkpoint shape change.

**Test cases (add to SS 8):**
- **8.19 gram no-op when OFF:** `gram_lambda=0` -> teacher not instantiated, loss/graph identical to baseline.
- **8.20 gram identity at teacher==student:** if teacher weights == student weights at step 0, `L_Gram == 0`
  (sanity: same volume through identical encoders gives identical Gram).
- **8.21 gram teacher frozen + off-optimiser:** teacher params have `requires_grad=False`, are absent from the
  optimiser param groups and from the student `state_dict`; a step does not update them.
- **8.22 gram bounds drift:** on a toy setup where the student is nudged toward collapse, `gram_lambda>0` keeps
  the student Gram closer to the teacher Gram (lower `||G_s-G_g||`) than `gram_lambda=0`.

## 4B. Training recipe — differential LRs + ViT warmup (restart from a healthy init)

**Motivation (Phase-0 diagnostic, step-0 vs ckpt-1000):** collapse is **induced**, not inherent. The pretrained
tower feeds the LLM **rank-12** tokens (post-proj eff_rank 11.9, sv[0]=0.24); ckpt-1000 feeds **rank ~1**
(post-proj eff_rank 1.07, sv[0]=0.96). Per-block norms grew 3-5x and per-block rank fell everywhere. The
uniform `--vit_lr 3e-4` over-cooks the *pretrained* blocks while the *new* parts legitimately need a higher LR.
Split them, warm up, and (for a fresh run) restart from the healthy init.

**Code changes required:**
- **Param groups (`swift/trainers/seq2seq_trainer.py` optimizer construction / `create_optimizer`):** today
  `--vit_lr` is a single group over all of `visual_3d`. Split `visual_3d` params into:
  - **pretrained ViT blocks** (Qwen tower blocks, excluding the new bits): LR **1e-5-3e-5** with **layer-wise
    decay** (`lr_i = base * decay^(depth - i)`).
  - **new / inflated params**: z-extended patch-embed (`conv_inflate`), z-RoPE, STA temporal params, projector:
    LR **1e-4-3e-4**.
  - Tag the new modules at attach time (a boolean attr, e.g. `_v3d_new=True`) so the group builder can separate
    them from the pretrained blocks; assign per-block LR via the layer index.
- **ViT warmup-freeze:** freeze the pretrained ViT blocks for the first `--vision_3d_vit_freeze_steps` (~300)
  steps, then unfreeze. Implement with a `TrainerCallback` that toggles `requires_grad`, OR keep the params in
  the optimizer and gate their LR to 0 via a `LambdaLR` per-group multiplier until the threshold (the LambdaLR
  route is safer with DeepSpeed — params stay tracked, only the effective LR is 0).
- **Zero-init projector output:** set the final `proj[-1]` Linear weight+bias to 0 so step-0 sends no gradient
  into the tower (volume tokens contribute nothing until the projector learns a sane map). Gate with
  `--vision_3d_proj_zero_init` (default `false` = current; recommended `true` for a fresh run).
- **Config:** set the group LRs, freeze steps, zero-init in the `.sh`.

**Backward compat:** if the new group args are unset, fall back to uniform `--vit_lr` (current behaviour);
`vit_freeze_steps=0` and `proj_zero_init=false` reproduce today exactly.

**Restart note (Phase-0 trajectory, post-proj centered eff_rank): step-0 = 11.9 -> ckpt-500 = 1.25 ->
ckpt-1000 = 1.07.** The collapse is essentially COMPLETE BY STEP 500 (sv[0]: 0.24 -> 0.89 -> 0.96). So neither
ckpt-500 nor ckpt-1000 is a healthy restart point — **only step-0 is.** Restart fresh, with anti-collapse
(VICReg + warmup + differential LR) active FROM STEP 0, since the damage lands in the first few hundred steps.
(blk23 norm is non-monotonic 1092 -> 195 -> 3488: the norm blow-up is a LATE symptom; the rank collapse is the
early root.)

## 4C. Monitoring — collapse detector + per-tag AUC (at eval cadence)

### (i) Collapse metric — **zero extra forward**
- Log the centered **effective rank** and **top-singular-value fraction** (`sv[0]`) of the **pre-proj / post-proj
  tokens that the training step already computed** — no zeroed-volume forward, no extra encoder pass. This tracks
  exactly the 12 -> 1 collapse Phase-0 measured. Compute the SVD on a **subsample** of tokens (a few hundred) to
  keep it trivial; log `metrics/postproj_eff_rank`, `metrics/postproj_sv0` every `logging_steps`. If VICReg (4A-b)
  is on, these are essentially free from the covariance already formed.
- **reldiff-proper (real vs zeroed volume) is NOT zero-overhead** — it needs an extra forward on a zeroed volume.
  So default to the rank/sv0 proxy above (respects the no-extra-compute constraint); only if reldiff-proper is
  wanted, run it at **eval cadence** on a tiny fixed probe (a few volumes, `no_grad`) — small but nonzero.
- Gated by `--log_collapse_metrics` (default `false`).

### (ii) Per-tag AUC on validation (cls_json only), whenever eval runs
- On each `evaluate()`, restricted to **cls_json** val rows (filter by the `channel` column == `cls_json`),
  compute per-tag **AUROC from the teacher-forced P(true)**: at each tag's boolean position in the roll-call JSON,
  `softmax` over `{true_id, false_id}` -> `P(true)`; label = the GT boolean. Accumulate per tag across the val
  set; log **macro-AUC** + per-tag AUC to wandb.
- **Reuses the presence-weighting machinery** already in `seq2seq_trainer.py`: runtime-derived `true_id`/`false_id`,
  the cls_json channel gate, and the boolean-token position mask. So most plumbing exists — the new part is the
  per-tag accumulation + `roc_auc_score`.
- **Overhead:** rides the teacher-forced eval forward that already runs; marginal cost is bookkeeping + a cheap
  AUC over accumulated scores, on the cls_json val subset only.
- Gated by `--eval_cls_auc` (default `false`). (The earlier standalone `eval_tag_auroc.py` OOM'd loading the
  encoder at max_tokens 4096 on L40S; wiring it into the training eval avoids the standalone load, but cap
  `--eval_cls_auc_max_tokens` if needed.)

## 4D. Encoder architecture — STA rank pressure + final-block massive-activation

### (i) STA temporal attention is a *rank-pressure* source — verified safe on residuals, but add a locality knob
- **Residual path is intact (good):** `STAVisionBlock` does `h = h + temp_attn(norm3(h))` with its own pre-norm,
  and `temp_attn.proj` is **zero-initialised** -> identity at step 0, gated residual preserved
  (`sta.py:64-65,124-135`). This is the STAVEQ2-style design the critique asks for; no change needed here.
- **But the temporal attention averages over the FULL z-extent:** `attn` is computed over all `gt x gt` temporal
  groups with **no locality** (`sta.py:99`). Attention is an averaging op; a large group applies stronger
  uniformity pressure than a local window. It trains up from zero exactly in the 0-500 window where rank
  collapsed — so STA is a *candidate driver*, not just a bystander.
- **Change:** add an optional **temporal window** `--vision_3d_sta_window k_t` (band-diagonal / local attention over
  `+/- k_t` groups; default `0` = full attention = current behaviour). **Ablate `k_t` against effective rank**
  (not just accuracy) — this directly tests the "large group accelerates collapse" hypothesis.
- **Diagnostic (cheap):** re-run the Phase-0 rank probe with STA disabled vs enabled on the SAME weights to
  quantify how much of the 12 -> ~2 per-block drop STA contributes.

### (ii) Final-block large norm — massive activation vs shared readout (decides the fix)
- **Sun et al. (2402.17762):** a few activations grow orders of magnitude larger than the rest, are largely
  input-invariant, and act as an implicit bias (present in ViTs incl. CLIP ViT-L). Clean fix: give the model
  **explicit learnable per-head `k'`, `v'` attention-bias parameters** — during fine-tuning it migrates the bias
  there and the massive activations disappear (register tokens do the same by another route).
- **RESULT (`channel_concentration.py`, blk23):** it IS a **massive activation** — `mu_PR = 1.2`, top-8 channels
  hold **97%** of the mean-vector energy — and it is **INHERENT**: present at step-0 (`||mu||=1113`) as well as
  ckpt-1000 (`||mu||=3521`). At step-0 the model is healthy (post-merger rank 12) *with* this massive activation,
  because the merger's `ln_q` normalises it away.
- **Decision: DEPRIORITISE `k'`/`v'` (do NOT implement `--vision_3d_attn_bias` now).** The massive activation is
  not what breaks us — step-0 has it and works fine. Adding `k'`/`v'` would not fix the rank collapse. Keep it as
  a documented later option only.
- **The real signal (blk20-22): a CORRELATION collapse, which points straight at VICReg's covariance term.**
  Per-channel variance is spread across MANY channels (`std_PR` 84->308 from step-0->ckpt-1000) yet the centered
  eff_rank FELL (2.8 -> 1.85): many channels vary but they became **linearly correlated**, so the variance lives
  in ~2 directions. That is exactly what the **covariance (decorrelation) term of VICReg (4A-b) targets** — not
  the variance floor, the *covariance* term. So 4A-b's covariance weight is the primary lever; raise its priority.

## 5. CLI args (backwards-compatible; default OFF)

Add to `swift/arguments/base_args/model_args.py`; thread template-visible ones through
`get_template_kwargs`/`get_template`/`Template.__init__` (same path used for `ct_normalize`).

| CLI arg | default | meaning |
|---|---|---|
| `--vision_3d_slowfast` | `false` | **master switch**; off => current behaviour exactly |
| `--vision_3d_fast_grid` | `"4,8,8"` | fast target grid `Tf,Hf,Wf` -> `fast_count = Tf*Hf*Wf` |
| `--vision_3d_fast_pool` | `avgmax` | `avg` \| `max` \| `avgmax` (avgmax => proj_fast in = `2*enc_dim`) |
| `--vision_3d_slow_max_tokens` | `4096` | cap on slow tokens (K/V memory) |
| `--vision_3d_crossattn_every_n` | `4` | hybrid-layer stride |
| `--vision_3d_crossattn_min_layer` | `0` | first eligible decoder layer |
| `--vision_3d_crossattn_max_layer` | `num_hidden_layers` | last eligible decoder layer |
| `--vision_3d_crossattn_heads` | `= num_key_value_heads` | cross-attn heads |
| `--vision_3d_crossattn_gate` | `warmup` | gate type (all zero-init) |
| `--vision_3d_crossattn_lr` | `= vit_lr` | LR for new params (optional) |
| `--vision_3d_proj_prenorm` | `false` | LayerNorm(enc_dim) before every projector; fixes baseline-swamping (**not** rank). off => `nn.Identity` |
| `--vision_3d_rankreg_lambda` | `0.0` | VICReg variance+covariance loss on pre-proj (slow) tokens; **raises effective rank**. 0 => off |
| `--vision_3d_rankreg_var_w` | `1.0` | weight of the variance (std-hinge) term |
| `--vision_3d_rankreg_cov_w` | `1.0` | weight of the covariance (decorrelation) term |
| `--vision_3d_rankreg_gamma` | `1.0` | target per-dim std for the variance hinge |
| `--vision_3d_rankreg_max_tokens` | `2048` | token subsample cap for the covariance term (cost control) |
| `--vision_3d_feature_layer` | `-1` | encoder block index feeding the projector; `-1` => last block (current) |
| `--vision_3d_gram_lambda` | `0.0` | Gram-anchoring loss weight (DINOv3); preserves dense/patch structure. 0 => teacher not built |
| `--vision_3d_gram_teacher` | `""` | path to the frozen Gram-teacher checkpoint (sharpest domain-adapted encoder) |
| `--vision_3d_gram_max_tokens` | `0` | cap M for the Gram matrix (0 => use all patch tokens) |
| `--vision_3d_vit_block_lr` | `= vit_lr` | LR for **pretrained** ViT blocks (recommend 1e-5-3e-5) |
| `--vision_3d_vit_lr_decay` | `1.0` | layer-wise LR decay for ViT blocks (1.0 => off) |
| `--vision_3d_new_param_lr` | `= vit_lr` | LR for **new** params (z-inflate, z-RoPE, STA, projector) |
| `--vision_3d_vit_freeze_steps` | `0` | freeze pretrained ViT blocks for the first N steps (warmup) |
| `--vision_3d_proj_zero_init` | `false` | zero-init projector output Linear (no grad into tower at step 0) |
| `--log_collapse_metrics` | `false` | log post-proj eff_rank + sv[0] from the existing forward (no extra pass) |
| `--eval_cls_auc` | `false` | per-tag AUROC on cls_json val from teacher-forced P(true), at each eval |
| `--eval_cls_auc_max_tokens` | `= vision_3d_max_tokens` | cap eval max_tokens for the AUC pass (memory) |
| `--vision_3d_sta_window` | `0` | temporal-attention window `k_t` (local +/- k_t groups); 0 => full z attention (current) |
| `--vision_3d_attn_bias` | `false` | learnable per-head k'/v' attention bias (ONLY if blk23 is a massive activation) |

---

## 6. Checkpoint save / load

**Save (all blocks incl. slow-fast):** cross-attn blocks (`kv_proj`, `q_proj`, `o_proj`, `gate`, `norm`) and
`proj_fast` are registered nn.Modules inside the decoder layers and `visual_3d` -> written to the model state_dict
by full-FT automatically (STA proves wrapped-layer submodules are saved). No extra save logic.

**Load (no compat/template issues):**
1. **Structure rebuilt before weights:** attach.py reads the saved `vision_3d_config`, sees `slowfast_enabled=true`
   + layer_idxs/dims, re-runs `install_slowfast` with the same layers/dims **before** the weight load -> no
   missing/unexpected keys (mirrors STA `sta_enabled` restore).
2. **Template restored:** `vision_3d_slowfast` + `fast_grid`/`fast_pool` threaded into template kwargs on reload
   (same path as `ct_normalize`), so the reloaded template does the fast/slow split + short `<video>` expansion.
3. **Dims/keys match** because structure is rebuilt from the saved config.

**Resuming a NON-slow-fast checkpoint (e.g. checkpoint-1000):** cross-attn params don't exist -> load base
`strict=False`, init new blocks fresh (gate=0). Effectively a new variant (warm base + fresh cross-attn) — an
explicit decision, not silent.

---

## 7. Dimension compatibility (validate at build)

| tensor | shape | check |
|---|---|---|
| encoder tokens/vol | `(M_i, enc_dim=2560)` | enc_dim from qwen_vl_token_dim |
| fast pooled (avgmax) | `(fast_count, 2*enc_dim)` | `fast_count = Tf*Hf*Wf` |
| fast projected | `(fast_count, 2560)` | proj_fast in = `2*enc_dim` iff avgmax |
| slow (batch) | `(B, T_slow, 2560)` | `T_slow <= slow_max_tokens` |
| cross-attn q | `(B, T_txt, n_heads, head_dim)` | `2560 % n_heads == 0` |
| cross-attn k/v | `(B, T_slow, n_kv_heads, head_dim)` | `n_heads % n_kv_heads == 0`, repeat_kv |
| gate | `(1,)` (+ optional `(B,T_txt,1)`) | zero-init |

---

## 8. Test cases (required)

| # | test | proves |
|---|---|---|
| 8.1 | **Dim/build**: construct `LLMCrossAttnBlock` with Qwen3.5 dims; assert sublayer shapes; assert raise on mismatched heads | dims line up |
| 8.2 | **Identity-at-init**: `SlowFastDecoderLayer` output == original layer output (gate=0) | starts as base model |
| 8.3 | **Fast avg+max**: assert `fast_count==Tf*Hf*Wf`, proj_fast in=`2*enc_dim`; a single hot voxel survives via max where pure-avg washes out | avg+max works |
| 8.4 | **Z coverage**: unique marker per z-slice -> every slice contributes to >=1 fast token; hot voxel on ONE slice appears in that window's MAX | avg+max reduces over z, no blind slices |
| 8.5 | **Splice length**: slow-fast on -> `<video>` expands to `fast_count`; masked_scatter fills exactly the fast positions | seq length + splice correct |
| 8.6 | **Conditioning/stash**: each hybrid layer holds `(B,T_slow,enc_dim)`+mask; non-hybrid untouched; cross-attn no shape error | plumbing correct |
| 8.7 | **Full-model forward**: one train step, slow-fast on -> finite loss, short seq, grads reach cross-attn params once gate!=0 | end-to-end runs |
| 8.8 | **Save/load round-trip**: save -> reload via attach.py -> state-dict key sets equal, forward outputs match | save/load OK |
| 8.9 | **Backwards-compat (golden)**: slowfast=false -> model, num_volume_tokens, forward IDENTICAL to pre-change | off => byte-identical |
| 8.10 | **Old-checkpoint load**: ckpt-1000 loads clean (off); loads base strict=False, new params init, gate==0 (on) | resume path defined |
| 8.11 | **Optimizer coverage**: new params in trainable/optimizer group (LR + grads) | not frozen/dropped |
| 8.12 | **GQA repeat_kv**: `n_kv_heads < n_heads` -> cross-attn output shape correct vs manual reference | GQA handled |

---

## 8A. Pre-launch failure-mode checklist (crash prevention)

Grouped by integration seam. Each line is a thing that can crash or silently no-op a run; check before launching.
A **1-step smoke** (forward+backward on a 2-sample batch, all flags ON) must pass, plus an **identity smoke**
(all new flags OFF -> loss bit-identical to current) before any full run.

**S1. Token-count / shape seams (slow-fast):**
- `num_volume_tokens()` returns `fast_count = Tf*Hf*Wf` **everywhere** `<video>` is expanded (template `_encode`,
  `_extend_tokens`, collator, `_splice_volume_embeds`). Any disagreement -> splice index error or
  `inputs_embeds`/labels length mismatch. Assert the count is identical in all three.
- `proj_fast` in-dim = `2*enc_dim` for `avgmax` (else `enc_dim`); out = `llm_hidden`. Wrong -> matmul shape error.
- Variable-resolution volumes: each has its own `(gt,gh,gw)`; the 3D adaptive pool must run **per volume** and
  still emit exactly `fast_count`. Test with two volumes of DIFFERENT sizes in one batch.
- `merge_size**2` accounting when deriving the fast grid from post-merger tokens (post-merger count already
  divided by `merge_size**2`). Off-by-`merge_size²` -> wrong reshape.
- Slow tokens: pad to `(B, T_slow_max, enc_dim)` + `slow_mask (B, T_slow_max)`; mask dtype/shape must match the
  cross-attn `key_padding_mask`. Test a batch where volumes have different slow-token counts (ragged -> padded).

**S2. Cross-attention into the LLM decoder:**
- `install_slowfast` must insert the hybrid layer at the correct decoder indices and match the REAL decoder
  layer `forward` signature (position_embeddings, attention_mask, past_key_values, cache_position, **kwargs) —
  a signature mismatch crashes on the first LLM forward. (Confirm against the mapped decoder class.)
- Cross-attn `kv_proj` maps `slow_dim(=enc_dim) -> kv`; head count = `num_key_value_heads`; head_dim consistent.
- Softmax in fp32 then cast back (mirror STA `sta.py:100`) to avoid bf16 overflow/NaN.
- Zero-init gate (and zero-init output proj) so step-0 output == base decoder layer output -> identity smoke.
- GatedDeltaNet (linear-attn) layers vs full-attn layers: only insert cross-attn on the intended set; verify the
  chosen indices aren't the linear-attn layers if those have a different forward contract.

**S3. Checkpoint save / load:**
- `vision_3d_config` records the slow-fast structure (enabled, layer_idxs, dims, fast_grid, fast_pool,
  slow_max_tokens). On resume the skeleton (`install_slowfast` + `proj_fast`) is rebuilt **before** `load_state_dict`
  or the new keys are "missing".
- Cross-attn params live inside the LLM decoder layers -> they ARE in `model.state_dict()` and get saved; verify
  their key names are STABLE across a save->reload (a rename breaks resume).
- Loading an OLD checkpoint (no slow-fast) with slow-fast ON -> `strict=False`, new params init, no error; with
  slow-fast OFF -> strict load clean.

**S4. Optimizer / DeepSpeed ZeRO-2 / LR groups:**
- Every NEW trainable param (`proj_fast`, cross-attn, STA already handled) is in **exactly one** optimizer group.
  A param in NO group = silently frozen (no gradient); in two = error. Enumerate `model.named_parameters()` with
  `requires_grad` and assert each maps to one group.
- `install_slowfast` + `proj_fast` build must run **before** DeepSpeed/optimizer init (attach runs pre-trainer) so
  ZeRO-2 tracks them. Verify ordering.
- Differential LRs (4B): partition {pretrained ViT blocks | new vision params | LLM} — assert the union covers all
  trainable params and the sets are disjoint.
- ViT warmup-freeze via LambdaLR: params STAY in the optimizer, LR multiplier 0 until `freeze_steps`; assert the
  multiplier actually returns to 1 at the threshold (a wrong closure keeps them frozen forever).
- Zero-init projector output: confirm it is the FINAL `proj[-1]` Linear, and that gradient still flows into it on
  step 1 (zero weight is fine; a zeroed input path would not).

**S5. VICReg / loss assembly:**
- `_rankreg_feats` (and `_slow_tokens`) stashed ONLY when `training and lambda>0`; every consumer is None-safe.
- Covariance term: subsample to `rankreg_max_tokens`, compute in fp32; guard `n<2` (no NaN), guard all-zero std.
- VICReg scalar added to the LM loss BEFORE `backward`; under **gradient accumulation** it must scale the same way
  as the LM loss (check swift's `num_items_in_batch` / accumulation handling so the term isn't N x too large).
- Presence-weighting + per-channel logging + VICReg must coexist (all hook `compute_loss`); run one step with ALL
  enabled and assert finite loss and non-zero grads on the new params.

**S6. Template / data / eval:**
- `--ct_augment false` (verified) — no corruption at train or eval.
- `CT_DATASET_CHANNELS=1` channel column still tagged; cls_json channel gate intact.
- `--eval_cls_auc`: the cls_json val subset is non-empty; `true_id`/`false_id` derive; boolean positions found;
  `roc_auc_score` skips tags with a single class present in the val slice (else it throws).
- `--log_collapse_metrics`: SVD on the subsampled tokens is wrapped (svd can fail to converge) -> try/except, log
  NaN rather than crash the step.

**S7. Memory / runtime:**
- Slow-token K/V memory = `crossattn_layers x B x T_slow_max x kv` — the main new memory cost; cap
  `slow_max_tokens` and verify peak on H100 at `per_device_train_batch_size`.
- Encoder at `max_tokens 4096` OOMed on L40S (STA attention) — training is H100, but the smoke should run at the
  real `max_tokens`, not 1024.
- Gradient checkpointing must wrap the new layers (inherit the flag, as STA does at `sta.py:163`).

**S8. Order-of-operations smoke sequence (run in this order):**
1. Import + build model with ALL new flags OFF -> forward+backward 1 step -> loss finite (baseline).
2. All flags ON -> build -> assert new params exist, in optimizer, `requires_grad=True`.
3. 1-step forward+backward, batch of 2 volumes of DIFFERENT sizes -> no shape error, loss finite, grads non-zero
   on `proj_fast` + cross-attn + (rankreg has no params but its loss term > 0).
4. Save checkpoint -> reload (skeleton rebuilt) -> forward matches pre-save (within bf16 tol).
5. Identity: flags OFF after having code present -> loss == step-1 baseline (bitwise).

## 9. Backwards compatibility (summary)
- `--vision_3d_slowfast false` (default): no split, no install, `num_volume_tokens` unchanged, current
  `_post_encode` path -> **byte-identical** model + state-dict + outputs (test 8.9).
- Saved config records `slowfast_enabled`; pre-existing checkpoints load exactly as before (test 8.10).

## 10. Build order
module Sec.1 + 8.1-8.2 -> split Sec.2 + 8.3-8.4 -> conditioning Sec.3 + 8.5-8.6 -> attach/args Sec.4-5 +
8.7,8.11,8.12 -> save/load Sec.6 + 8.8,8.10 -> backwards-compat 8.9.

## 11. Open decisions
1. RoPE on cross-attn query (none proposed vs text-position on q).
2. Which layers to hybridise (every-N vs bias to the 8 full-attn layers vs GatedDeltaNet layers).
3. Fast grid size (4x8x8=256 vs larger).
4. Slow cap (full M vs slow_max_tokens; K/V mem = T_slow * n_kv_heads * head_dim * 2 * n_hybrid_layers).
5. Warm base + fresh cross-attn (resume strict=False) vs fresh full run.

## 12. Non-goals / caveats
- **Encoder cost unchanged** — slow tokens come from the full encoder pass; this shortens the LLM/sequence, not the
  vision tower (which OOMs at 4096). Encoder relief needs separate work (vision-tower flash-attn / lower slow res).
- Presence-weighting / per-channel logging / AUROC tooling unaffected (they key on response tokens).

---

## 9. FAST-CHANNEL COLLAPSE — diagnosis + fix (added 2026-10-02)

**Observation (reldiff/rank probe across ckpt-200/400/600 of the 4B slow-fast run):** the PRE-proj encoder
representation is healthy and enriching (post_merger eff_rank 11.7 → 16.0 → **41.8**, reldiff ~1.2, fully
input-dependent), but the POST-proj FAST tokens (the ones spliced into `<video>`) **collapse**:

| ckpt | post_proj ‖feat‖ | post_proj eff_rank | post_proj reldiff |
|---|---|---|---|
| 200 | 0.10 | 1.41 | 0.52 |
| 400 | 0.60 | 1.41 | 0.52 |
| 600 | **4.3** | **1.34** | **0.31** |

i.e. the fast tokens' magnitude explodes while they stay **rank ~1** and become *relatively less*
input-dependent (reldiff dropping). Free-running CAC stayed all-zeros at 200/400/600 as a result — the fast
`<video>` channel delivers a strong but low-rank, increasingly input-INDEPENDENT signal (a learned bias),
not volume-discriminative content.

**Root cause:** VICReg rank-reg regularises only the PRE-proj encoder tokens (`_rankreg_feats`), keeping the
encoder rich. **Nothing regularised the post-proj fast tokens**, so `fast_pool_3d` + `proj_fast` are free to
squash the rank-42 features into a rank-1 output (and `proj_zero_init` on `proj_fast` seeds a single dominant
direction).

**Fixes implemented (behind flags; all default off = prior behaviour):**
1. **Post-proj VICReg** on the fast tokens — `--vision_3d_rankreg_fast_lambda` (backbone stashes
   `_rankreg_fast_feats = cat(fast_out)` in `_slowfast_forward`; trainer adds `rankreg_loss` on it over the
   fast-token axis). Directly penalises the rank-1 collapse. **KEEP pre-proj VICReg too — it protects the
   encoder; this is IN ADDITION, not instead.**
2. **Orthogonality reg** on `proj_fast` Linears — `--vision_3d_proj_ortho_lambda` (`ortho_reg` = `||G-I||²/d²`
   in slowfast.py). Keeps the projection full-rank in weight space and bounds its singular values (curbs the
   norm blow-up).
3. **Weight decay (L2) on `proj_fast`** — `--vision_3d_proj_fast_wd` (loss-space L2 on proj_fast weights).
4. **`proj_fast` init** — `--vision_3d_proj_fast_init orthogonal` (full-rank start instead of the rank-1-seeding
   zero init; base/slow `proj` still zero-inits via `--vision_3d_proj_zero_init`).

Also for the 2B run: **fast tokens 512 → 1024** (`--vision_3d_fast_grid 16,8,8`, doubling z-resolution) since
the slow branch now carries 12k tokens.

**Tests:** `tests/vision_3d/test_fast_collapse_fixes.py` (ortho_reg 0-on-orthonormal / large-on-rank-1 /
scale-free / rectangular; proj_fast orthogonal init full-rank; rankreg penalises a collapsed fast batch
0.99 vs 0.01). Existing slow-fast suites remain green.

**CAUTION (symptom vs disease):** rank-reg makes the fast tokens *diverse* but won't force the LLM to *use*
them. Pair with a fast-channel-usage ablation (zero the fast tokens, measure LM-loss delta) — if the LLM
ignores `<video>` (relying on slow cross-attn + priors), the real fix is making the channel count, not just
high-rank.
