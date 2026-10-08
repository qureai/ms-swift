# Curia (Radiology DINOv2) Integration Plan

Goal: use **Curia** (raidium-med/curia) as a vision encoder with the **Qwen3.5-4B LLM**.
- **Phase 1:** Curia in the **2D ViT slot** + Qwen LLM, **no vision_3d**. CTs processed **slice-by-slice** (Curia is 2D). Evaluate how a 2D radiology-pretrained encoder does vs. our 3D-adapted Qwen tower.
- **Phase 2:** if Phase 1 is good — run **Curia (2D) AND the STA-adapted Qwen encoder (3D) side-by-side**.

STATUS: planning only, **no code changes yet**. Some facts need verifying on access (see §7).

---

## 1. What Curia is (verified from repo / HF / arXiv 2509.06830)

- **Encoder ONLY — no LLM.** DINOv2-adapted ViT, self-supervised on ~150k CT/MRI exams (130 TB). `AutoModel.from_pretrained("raidium/curia", trust_remote_code=True)` → feature embeddings (CLS + patch tokens). `AutoModelForImageClassification` just adds task heads. No text decoder.
- Variants: **Curia-B** (ViT-B, 86M; ~768-dim, 12 layers) and **Curia-L** (ViT-L, 300M; ~1024-dim, 24 layers). HF hosts the base.
- **Patch 16×16**, input resized to **512×512**, **grayscale (1 channel)**, **z-score normalized, NO windowing**.
- **Fundamentally 2D**: paper says *"Curia is fundamentally a 2D model, processing volumetric CT and MRI data on an image-by-image basis."* It runs the ViT per slice, then aggregates slice tokens by **mean-pool or a learned cross-attention** — **no native 3D/z-axis attention**.
- Per slice at 512/16 = **32×32 = 1024 patch tokens** + 1 CLS, 768-dim (ViT-B).
- Model is **GATED** on HF (config.json returned 401) → needs license acceptance + an HF token.

---

## 2. Two integration routes

### Route B — pluggable encoder path (RECOMMENDED for Phase 1)
Reuse the existing `vision_3d` machinery (`attach.py` / adapter / `Vision3DBackbone` / `_splice_volume_embeds`) with a **2D Curia adapter** and **STA off**. Curia attaches as an *additional* encoder (the `visual_3d` slot). This is "Curia 2D encoder + Qwen LLM" with the splice/projector/save-load we already have working. The `<video>` placeholder is just the carrier; "2D, no z-mixing" = STA disabled.
- Pros: minimal new surface, reuses tested splice + resume-safe save/load + optimizer grouping; Phase-2 side-by-side falls out naturally.
- Cons: conceptually routes a 2D encoder through the "3d" plumbing (naming only).
- **Native 2D tower stays loaded + frozen (confirmed):** the host Qwen3.5-VL keeps its native `model.visual` tower. The current 30-Sep run already freezes it (`freeze_parameters: ['model.visual', 'model.visual.merger']`) and never invokes it for CT (no `<image>` inputs). Under Route B the same holds — Curia does the real work, `model.visual` sits frozen + unused (~1–2 GB/GPU dead weight). Harmless but wasteful.
  - **Route B + prune (clean variant):** after load, delete/stub `model.visual` to reclaim the memory. Precedent: the `vision_3d` loader already "releases non-vision parts of the source model." Gives a clean Curia-only vision stack. Do this if GPU memory is tight.

### Route A — replace Qwen-VL's native `model.visual` 2D tower
Literally swap `Qwen3_5VisionModel` for Curia.
- Cons: Qwen-VL's visual tower is **deeply coupled** into the model forward (its merger → `out_hidden_size`, `image_grid_thw`, deepstack, 2D rope, dynamic-resolution processor). Curia has none of that (plain DINOv2, fixed 512, 768-dim, no merger). A clean swap needs a Curia wrapper that mimics the `visual(pixel_values, grid_thw) -> (N, out_hidden_size)` contract **and** a replaced image processor. Higher risk, more coupling, little benefit over Route B.
- Verdict: avoid unless we specifically need Curia on the native `<image>` path.

---

## 3. Changes for Route B (Phase 1)

1. **New `CuriaAdapter`** (`swift/model/vision_3d/curia.py`, analogous to `QwenVLVideoAdapter`):
   - Input: a volume `(T, C, H, W)`. For each slice: resize → **512×512**, **z-score**, **1 channel** (see channel decision below).
   - Run Curia **batched over the T slices** (`(T, 1, 512, 512)` → DINOv2 forward) → per-slice tokens `(1024, 768)` (+ CLS).
   - Emit either: (a) **patch tokens** `(T × 1024, 768)` (spatial detail, large — needs pooling), or (b) **CLS per slice** `(T, 768)` (compact, 1 token/slice), or (c) pooled patch tokens per slice. Start with a config knob; CLS-per-slice is the cheapest baseline.
   - No `grid_thw`/merger/STA.
2. **Loader** (`loader.py`): load `raidium/curia` via `AutoModel` + `trust_remote_code=True` + **HF token** (gated). Extract the ViT module. Confirm output keys (`last_hidden_state` / `pooler_output` / custom `features`) — verify on access (§7).
3. **Preprocessing** (template `load_volume` / a Curia branch): Curia-specific path — **bypass `--ct_windows`**, use the **as-trained** 1-channel z-score at 512×512 (NO windowing). This is a real divergence from our multi-window CT pipeline (see channel decision).
4. **`enc_dim = 768`** → projector `Linear(768 → llm_hidden=2560)` (the backbone builds `proj` from `enc_dim`, so this is automatic once `enc_dim` is set).
5. **STA off** (`--vision_3d_sta false`) and **slow-fast off** initially (plain projector path). Token budget: 1024/slice × T is large; reuse the existing pooling / `vision_3d_max_tokens` cap, or use CLS-per-slice.
6. **Config/args**: a `--vision_3d_model raidium/curia`, an encoder-type switch (qwen-vl vs curia) so `attach.py` picks `CuriaAdapter`, `--vision_3d_trust_remote_code true`, HF token env.

### Channel decision (the key "use as-is" choice)
Curia was trained **1-channel grayscale, z-scored, no windowing**. Our CT pipeline uses **11 window channels**. To use Curia **as-is** (keep its pretrained features valid), feed it **one grayscale window** (e.g. a full-range or soft-tissue window), z-scored, 512×512 — do NOT inflate its patch-embed to 11 channels (that would push it OOD and forfeit the pretraining). This means the Curia path runs a different CT preprocessing than the Qwen path. Document which single window we pick (ablatable).

---

## 4. Phase 2 — Curia (2D) + Qwen-3D encoder side-by-side

Two encoders at once: Curia on the per-slice 2D features, the STA-adapted Qwen tower on the 3D path. Our system currently supports **one** secondary encoder (`model.visual_3d`). Side-by-side needs **two attach points**, e.g.:
- Curia on the native `<image>` channel (Route A-style) OR a second `visual_3d`-like slot, projecting 768→2560; and
- the Qwen-3D encoder on the `<video>` channel (current `vision_3d`), projecting 2560→2560.
Both splice into `inputs_embeds` at their own placeholder ids; both get their own optimizer group. This is a meaningful extension (a second encoder slot + two placeholder routings) — scope it only after Phase 1 shows Curia helps.

---

## 4b. Why NOT Curia + Qwen *text* LLM + a from-scratch merger
Tempting idea: load Curia + the Qwen3.5 **text** LLM (not the VLM) and train only a merger/projector.
- **Possible** (LLaVA-style connector), but the text LLM is **vision-naive** — it has no machinery to use vision tokens. A trained merger with a **frozen** LLM gets only partway (LLaVA result); you'd have to **train the LLM** on substantial vision-language data to teach it to consume vision tokens. CT is limited-data + specialized → steep.
- The Qwen **VLM**'s `language_model` is **already co-trained with vision** (it consumes the native visual tower). Keeping it means inheriting that pretrained vision-consumption ability; adapting it to Curia + CT needs far less data. Dropping to the text LLM throws that away.
- **Conclusion:** keep the VLM's (vision-adapted) LLM and attach Curia as the encoder (Route B) — you get a vision-capable LLM AND Curia features, and we already train the projector + LLM. Pruning the dead `model.visual` tower (§2) does NOT remove the LLM's vision ability — that lives in `language_model`, which stays.

## 5. What Phase 1 actually tests
Whether a **2D radiology-pretrained** encoder (Curia, slice-by-slice, no z-mixing) gives the LLM better CT grounding than our **3D-adapted natural-image** Qwen tower. Compare against the current 30-Sep run's grounding/AUC baseline on the same val/test. If Curia's per-slice features win, Phase 2 (both encoders) is justified.

---

## 6. Effort estimate (Route B, Phase 1)
- `CuriaAdapter` + loader (AutoModel/trust_remote_code/gate): moderate.
- Curia preprocessing branch (512 z-score grayscale, single window): moderate (new path in `load_volume`).
- enc_dim/projector/token-budget wiring: small (reuses existing).
- Config args + encoder-type switch in `attach.py`: small.
- Smoke tests (same §8A discipline as slow-fast): the usual.

---

## 7. Facts verified ON ACCESS (HF access granted 01-Oct-2026, `raidium/curia`)
Resolved from `config.json`, `preprocessor_config.json`, `modeling_dinov2.py`, `curia_image_processor.py`:
- **Encoder = `Dinov2Model` ViT-B**: `hidden_size 768`, `num_hidden_layers 12`, `num_attention_heads 12`, `patch_size 16`, `image_size 512`, **`num_channels 1`**, `use_swiglu_ffn true`, layerscale 1e-5, `apply_layernorm true`, `drop_path_rate 0.4`. → per slice **32×32 = 1024 patch tokens + 1 CLS, 768-dim**. (HF hosts only the **base/768**; no ViT-L on the repo → use Curia-B.)
- **Output**: `AutoModel` → `BaseModelOutputWithPooling` → `last_hidden_state` (patch tokens) + `pooler_output` (CLS, layer-normed since `apply_layernorm`). So **`enc_dim=768`**, projector `768→2560` is automatic.
- **NO register tokens** (plain `Dinov2Model`; `use_mask_token true` but no registers) — the DINOv2-with-registers worry is closed. Confirm CLS-vs-patch token ordering at first load (trivial).
- **Preprocessing (`CuriaImageProcessor`, `auto_map` → `trust_remote_code`)**: resize to **512 BICUBIC antialias** → **per-image z-score** (channel-wise `NormalizeIntensity`, `eps 1e-6`) → **NO windowing**; `clip_below_air=false` (no HU clip by default). This confirms the §3 channel decision: feed **one grayscale window, z-scored, 512, 1-channel** — do NOT inflate to 11 channels.
- **Bonus**: `modeling_dinov2.py` ships a learned **cross-attention slice-aggregation head** (`AttentionConfig`: `num_queries`, self+cross blocks) — Curia's own volume pooling. Phase-1 bypasses it (feed per-slice tokens to our projector); available if we later want their pooling.
- Repo also hosts ~17 downstream task heads (`AutoModelForImageClassification` subfolders: `anatomy-ct`, `kits`, `ich`, …) — not needed.
- Still TODO at runtime: pick which single window to feed (ablatable); confirm token ordering on first load.

---

## 7b. Backward compatibility with the existing Qwen-3D tower (REQUIREMENT)
The Qwen-3D path (STA-adapted Qwen tower + slow-fast + VICReg + split-LR, the current 30-Sep run) must be **byte-identical** after Curia lands. This is guaranteed by **additive, type-dispatched** changes — never mutating the Qwen path:
- `attach.py` already dispatches adapter by tower type (`atlas` / `qwen_vl` / generic, lines 149-186). Curia = a **new `elif is_curia_tower(tower)` branch**; the `qwen_vl` branch is untouched. Existing configs point `--vision_3d_model` at Qwen → `is_qwen_vl_tower` → identical path.
- **STA is already guarded to `qwen_vl` only** (attach.py:193, raises otherwise). Curia runs `--vision_3d_sta false` → no trip, no change to Qwen STA.
- slow-fast / `split_lr_groups` / rankreg / zero-init projector are generic (keyed on `enc_dim`/params); Phase-1 leaves them OFF for Curia; the Qwen behavior is unchanged.
- Shared files touched **additively only**: `loader.py` (Curia `AutoModel`+`trust_remote_code` load branch, keyed on model id), `attach.py` (`elif` + `CuriaAdapter`), `backbone.py` (`CuriaAdapter` is a NEW class on the existing adapter contract — `QwenVLVideoAdapter` not edited), and the template `load_volume` (NEW Curia preprocessing branch — **the one seam needing care**: gated on encoder type so the default multi-window CT path is byte-identical).
- New CLI args **default to current behavior** → existing `.sh` configs unchanged.
- Add a **regression test**: attaching the Qwen-3D encoder produces the same module tree + saved `vision_3d` config as before (guards against accidental shared-path edits).

---

## 8. Recommendation
Do **Route B, Phase 1** first: Curia via the pluggable path, STA/slow-fast off, 1-channel z-score 512 preprocessing, CLS-or-pooled patch tokens → projector → Qwen LLM. It's the lowest-risk way to measure Curia's slice-by-slice performance and reuses everything already working. Defer the native-slot swap (Route A) and the side-by-side (Phase 2) until Phase 1 data is in.

---

## 9. Test cases (required)

Same discipline as the slow-fast §8/§8A plan: a **1-step smoke** (forward+backward, Curia path, all Curia flags
ON) must pass, PLUS — and this is the headline requirement — a **Qwen golden/identity smoke** proving the existing
Qwen-3D path is **byte-identical** after the Curia code lands. Test ids `C.*`. **Group A is the backward-compat
firewall and is non-negotiable before any Curia run.**

### Group A — Backward-compatibility firewall (Qwen-3D path must not move)
| # | test | proves |
|---|---|---|
| C.1 | **Golden attach (Qwen)**: with NO Curia args, `attach_vision_3d_encoder` on `Qwen/Qwen3.5-4B` produces the SAME module tree, `enc_dim`, adapter_type=`qwen_vl`, and saved `vision_3d` config dict as pre-change (deep-equal) | attach path untouched |
| C.2 | **Golden forward (Qwen)**: 1-step smoke with the CURRENT 30-Sep `.sh` (Qwen tower, STA, slow-fast, VICReg) → loss, grads, token count **bit-identical** to the pre-Curia commit | the live run is unaffected |
| C.3 | **Golden volume preprocessing (CRITICAL)**: for a fixed CT volume, the Qwen path's `pixel_values` tensor (11-window, our resize/normalize) is **byte-identical** before and after the Curia preprocessing branch exists. The Curia branch is reached ONLY when encoder-type==curia | **volume processing change cannot touch Qwen runs** |
| C.4 | **Dispatch disjointness**: `is_curia_tower()` is TRUE for Curia, FALSE for the Qwen tower; `is_qwen_vl_tower()` is the reverse. No tower matches both; a Qwen config never enters a Curia branch | type dispatch is clean |
| C.5 | **STA guard holds**: Curia config with `--vision_3d_sta false` → `install_sta` NOT called, no raise; and the existing guard still RAISES if someone sets `sta true` on a non-qwen adapter | STA stays Qwen-only |
| C.6 | **Args default to Qwen behaviour**: every new Curia arg unset → identical argparse namespace effect as today (encoder-type defaults to qwen/auto) | old `.sh` configs unchanged |
| C.7 | **Cross-load refusal**: loading a Curia checkpoint (enc_dim 768) under a Qwen config (enc_dim 2560) fails loudly / `strict=False` with explicit missing-key report — never a silent shape clobber | checkpoints don't cross-contaminate |

### Group B — Curia encoder correctness
| # | test | proves |
|---|---|---|
| C.8 | **Loader**: `AutoModel.from_pretrained("raidium/curia", trust_remote_code=True)` + HF token loads the `Dinov2Model`; `enc_dim` inferred == 768; `trust_remote_code` is **per-encoder** (Curia true, Qwen stays false) | gated load works, no global flag leak |
| C.9 | **Output contract**: forward a `(T,1,512,512)` batch → `BaseModelOutputWithPooling`; `last_hidden_state` is `(T,1025,768)`, `pooler_output` `(T,768)`; confirm CLS-vs-patch **token ordering** (CLS index) empirically | we select the right tokens |
| C.10 | **Adapter shape**: `CuriaAdapter` turns a volume `(T,C,H,W)` into tokens `(N,768)` in the configured mode — (a) patch `T×1024`, (b) CLS-per-slice `T`, (c) pooled patch — each emits the asserted `N` | adapter contract honoured |
| C.11 | **Projector auto-build**: `enc_dim=768` → projector `Linear(768→2560)` built automatically; in-dim asserted | 768→llm_hidden wiring |
| C.12 | **1-channel enforced**: patch-embed expects `num_channels==1`; feeding 3 or 11 channels raises a clear shape error (NOT silent broadcast) | no accidental channel inflation |
| C.13 | **bf16 numerics**: Curia (config `torch_dtype float32`) run under bf16 encoder forward stays finite and close (rtol) to fp32 on a probe batch; projector matches LLM dtype | dtype policy safe |

### Group C — Curia preprocessing correctness (the risky shared seam)
| # | test | proves |
|---|---|---|
| C.14 | **Curia transform matches repo**: our Curia branch reproduces `CuriaImageProcessor` — resize **512 BICUBIC antialias** → per-image channel-wise z-score (`eps 1e-6`) → `clip_below_air=false` (no HU clip). Output per slice mean≈0, std≈1 | as-trained preprocessing |
| C.15 | **Single-window, not 11**: the Curia branch feeds exactly ONE grayscale window (the configured one), shape `(T,1,512,512)` — never the 11-window stack | channel decision enforced |
| C.16 | **Interpolation parity**: BICUBIC+antialias used (not our default bilinear) — mismatched interp flagged; ideally call `CuriaImageProcessor` directly to remove divergence risk | no silent distribution shift |

### Group D — Save/load, optimizer, memory
| # | test | proves |
|---|---|---|
| C.17 | **Resume skeleton**: `vision_3d_config` records `encoder_type=curia`, `enc_dim 768`, preprocessing mode, `trust_remote_code`; on resume the Curia skeleton + projector rebuild **before** `load_state_dict` | resume path defined |
| C.18 | **Encoder trainable/frozen policy**: decide frozen (exclude from checkpoint + `requires_grad False`, reload from HF) vs full-FT (saved, in optimizer). Test asserts the chosen policy: every Curia param is in **exactly one** optimizer group OR cleanly frozen | no silently-frozen / double-grouped params |
| C.19 | **LR-group name matching**: `split_lr_groups` keys off Qwen module names (`patch_embed`, `blocks`); Curia's naming differs → assert Curia params are NOT silently dropped from groups and the aligner group captures the projector | differential-LR wiring survives a different encoder |
| C.20 | **ZeRO-2 ordering**: Curia build + projector run before DeepSpeed/optimizer init (attach is pre-trainer) → ZeRO-2 tracks them | params tracked |
| C.21 | **Slice batching + token budget**: a T-slice volume batches through ViT-B without OOM; `T×1024` capped via pooling/`vision_3d_max_tokens`/CLS-mode; the emitted token count equals what the `<video>` splice expects | memory + splice length correct |

---

## 9A. Pre-launch failure-mode checklist (crash / silent-no-op prevention)

Grouped by integration seam, slow-fast §8A style. Each line can crash or silently no-op a run — check before launch.

**C-S1. Backward-compat firewall (HIGHEST PRIORITY — Qwen runs must not move):**
- Volume-preprocessing dispatch is keyed on **detected encoder type**, and the Curia branch is *unreachable*
  from a Qwen config. Prove with C.3 (byte-identical Qwen `pixel_values`). A shared mutable default (e.g. changing
  `load_volume`'s signature/window default) is the classic way to break this — avoid; branch, don't rewire.
- No new Curia arg changes a Qwen run's namespace/behaviour when unset (C.6). Encoder-type defaults to qwen/auto.
- `is_curia_tower` / `is_qwen_vl_tower` are mutually exclusive (C.4) — a Qwen tower must never enter a Curia path.
- Run C.2 (current 30-Sep `.sh` smoke) on the Curia branch and diff loss against the pre-Curia commit — **gate the
  merge on bit-identical**.

**C-S2. Loader (gated, trust_remote_code):**
- HF gated access: license accepted + token present on the training nodes (`HF_TOKEN` / `huggingface-cli login`).
- `trust_remote_code=True` for Curia ONLY — must not flip the Qwen load (which runs `false`). Per-encoder flag.
- `AutoImageProcessor` auto_map (`curia_image_processor.CuriaImageProcessor`) resolves, or we deliberately bypass it
  and replicate the transform (C.14). Decide one; don't half-load it.
- Extract the ViT module cleanly (the repo also ships task heads + a cross-attn aggregation head — load only the
  `Dinov2Model` encoder for Phase 1).

**C-S3. Token / shape seams:**
- `num_volume_tokens()` returns the Curia token count (patch `T×1024`, or CLS `T`, or pooled) **everywhere** the
  `<video>` placeholder is expanded (template `_encode`, `_extend_tokens`, collator, `_splice_volume_embeds`).
  Disagreement → splice index / `inputs_embeds`-labels length mismatch.
- Projector in-dim = **768** (not 2560). A stale `enc_dim` → matmul shape error.
- Per-slice: `(T,H,W)` → `(T,1,512,512)` → encoder → `(T, 1024 or 1, 768)`. Assert the reshape and the CLS/patch
  selection. Variable T across a batch → ragged token counts → pad + mask (reuse the slow-fast padding discipline).
- 1-channel hard requirement (C.12): never feed 3/11 channels into Curia's patch-embed.

**C-S4. STA / slow-fast OFF for Phase 1:**
- `--vision_3d_sta false` and slow-fast off → plain mean-pool/CLS + projector path; assert `install_sta` and
  `install_slowfast` are NOT called for the Curia adapter; the STA guard (qwen_vl-only) does not raise.

**C-S5. Checkpoint save / load:**
- `vision_3d_config` records encoder_type/enc_dim/preprocessing/trust_remote_code; resume rebuilds the Curia
  skeleton before `load_state_dict` (C.17).
- Encoder frozen→excluded-from-ckpt (reload from HF, pin the revision) vs trainable→saved; pick and enforce (C.18).
  If frozen, pin the HF commit hash so a re-pull can't silently change weights mid-project.
- Cross-config load refusal (C.7): a 768-dim Curia state must never silently load into a 2560 Qwen slot.

**C-S6. Optimizer / DeepSpeed ZeRO-2 / LR groups:**
- Every trainable Curia param in **exactly one** group (C.18); frozen params `requires_grad False` + out of groups.
- `split_lr_groups` name-matching assumes Qwen module names — verify it still captures the Curia projector and does
  not silently drop Curia encoder params (C.19). For Phase 1, simplest is: Curia encoder frozen, only projector
  trains (one aligner group).
- Build order: Curia attach + projector before optimizer/DeepSpeed init (C.20).

**C-S7. Preprocessing (shared seam, see C-S1):**
- BICUBIC + antialias (not bilinear); per-image channel-wise z-score, eps 1e-6, no HU clip (C.14–C.16).
- Pick the single window (full_range or soft_tissue) and record it; it's ablatable, not a silent default.

**C-S8. Memory / runtime:**
- Slice-batch the T-slice forward (don't push all T at once if T is large); cap the token budget (`T×1024` can be
  huge) via pooling / `vision_3d_max_tokens` / CLS-per-slice (C.21).
- Curia config is `torch_dtype float32`; running it fp32 ~doubles encoder memory — default to bf16 cast and verify
  numerics (C.13).
- Smoke at the REAL `max_tokens` and real slice count, not a toy depth.

### Phase-2 (side-by-side) — deferred tests, listed for completeness
Two encoders at once (Curia on one slot + Qwen-3D on `<video>`) adds: two placeholder-id routings must not collide;
two projectors (768→2560 and 2560→2560) to distinct token streams; two optimizer groups; the splice must fill each
placeholder from the correct encoder. Scope these only after Phase-1 shows Curia helps.
