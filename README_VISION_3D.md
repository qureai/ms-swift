# 3D vision encoder (CT volumes) for ms-swift

This fork adds an **optional secondary 3D vision encoder** to any multimodal LLM, so a VLM can consume
**3D CT volumes** alongside its normal 2D images. The 2D encoder is the VLM's built-in vision tower
(loaded with `--model`); the 3D encoder is a separate model you load with `--vision_3d_model`. CT
volumes ride the existing `<video>` token (no new special token) and are routed to the 3D encoder,
while images keep going to the 2D encoder.

- **Images** → `model.visual` (the host VLM's 2D tower)
- **CT volumes** (`<video>` pointing at a `.nii/.nii.gz/.npy/.npz/.safetensors`) → `model.visual_3d`
- Both can appear in the same sample; the absent modality is handled gracefully.

Supported 3D encoders (auto-detected from the model you pass):
- **Atlas / Pillar0** (`YalaLab/Pillar0-*`, `clip_multimodal_atlas`) — natively 3D.
- **Any Qwen2/2.5/3-VL vision tower** — Photon-style: the volume is patchified as a "video".
- **Any simple 3D CNN** encoder (tensor-in → features-out).

Supported host templates: `qwen2_5_vl_ct` (Qwen2.5-VL **and Qwen3.5**, which reuse this template) and
`qwen3_vl_ct` (Qwen3-VL). A common setup is a Qwen3.5-4B host whose **own** vision tower is loaded as the
3D encoder (`--vision_3d_model` = the same model) — the 2D and 3D encoders share an architecture but are
separate instances (the 3D one is trainable; the 2D one is typically frozen).

---

## New arguments

### 3D-encoder (model) arguments
| Argument | Default | Description |
|---|---|---|
| `--vision_3d_model` | `None` | Model id/path whose vision tower is used as the 3D backbone. `None` ⇒ feature off (behaves like upstream ms-swift). May be a full VLM (only its vision tower is extracted) or a dedicated 3D encoder (e.g. Pillar0). |
| `--vision_3d_trust_remote_code` | `False` | Pass `trust_remote_code=True` when loading the 3D encoder. Required for custom-code models such as `YalaLab/Pillar0-ChestCT`. |
| `--vision_3d_module_path` | `None` | Explicit dotted attribute path to the vision tower inside the loaded model (overrides the arch registry / class-name heuristic), e.g. `visual` or `model.vision_model`. |
| `--vision_3d_inflate_weights` | `True` | When an extracted tower has a 2D `Conv2d` patch embed, replace it with a `Conv3d` and inflate the pretrained 2D weights across the new temporal axis (vs. random init). |
| `--vision_3d_max_tokens` | `None` | **Required for CT training.** Fixed mode: number of tokens each volume expands to (the adapter pools the encoder output to this many). Variable mode (see below): the **maximum** tokens allowed per volume (the smart-resize ceiling). Larger = richer 3D representation but longer sequences. |
| `--vision_3d_resampler` | `none` | Token aggregator: `none` = blind mean-pool to `max_tokens` + 2-layer MLP projector; `perceiver` = learned Perceiver resampler (latent queries cross-attend to the encoder tokens, emit exactly `max_tokens`). Perceiver is **incompatible with variable-resolution** (it is fixed-length). |
| `--vision_3d_resampler_depth` / `--vision_3d_resampler_heads` | `2` / `8` | Perceiver cross-attention layers / heads (only used when `--vision_3d_resampler perceiver`). |
| `--vision_3d_sta` | `False` | Inject **Spatial-Temporal Attention** (temporal/z-axis self-attention) into each Qwen-VL vision block. Off = feature disabled (identical to upstream). Qwen-VL 3D encoder only (Atlas is natively 3D). See the STA section below. |
| `--vision_3d_sta_num_temporal_heads` | `None` | Temporal-attention heads. `None` ⇒ `round(spatial_heads × 0.25)` (4× fewer heads, same head_dim). |

### CT-volume (data / template) arguments
| Argument | Default | Description |
|---|---|---|
| `--template` | — | Use `qwen2_5_vl_ct` or `qwen3_vl_ct` to enable the CT template for the corresponding host. |
| `--ct_windows` | `[]` | Anatomical HU windows added as extra channels (comma- or space-separated). Valid: `lung, mediastinum, abdomen, liver, bone, brain, subdural, stroke, temporal_bone, soft_tissue`. E.g. `--ct_windows lung,mediastinum,bone`. |
| `--ct_window_base` | `full_range` | Base channel prepended to `ct_windows`: `full_range` (= HU `[-1000,1000]`), `minmax` (per-volume min-max), or `none`. |
| `--ct_volume_size` | `None` | Resize target `D,H,W` (each volume is trilinear-resized to this). **If unset ⇒ variable-resolution mode** (per-volume native-ish size, capped at `--vision_3d_max_tokens`) — see the Variable-resolution section. See also the encoder-specific constraints below. |
| `--ct_augment` | `False` | Enable Photon-style probabilistic volume augmentation during training (temporal/spatial shuffles, masking, noise, blackout). |
| `--ct_augment_prob` | `0.15` | Probability of applying an augmentation to a volume (when `--ct_augment` is on). |

**Channel count = 3D encoder input channels.** The number of channels produced is
`len(ct_windows) + (0 if ct_window_base==none else 1)`, and it must equal the 3D encoder's first-conv
`in_channels`. If they differ, the patch embed is **reinitialised** to the windowing channel count
(the rest of the encoder keeps its pretrained weights). To keep a pretrained patch embed, match the
window count to the encoder's native channels (Pillar0-ChestCT = **11**; Qwen-VL = **3**).

> **Patch-embed `in_channels` sync (Qwen towers):** a Qwen `*VisionPatchEmbed` stores `in_channels`
> as its **own attribute** and uses it to `view(-1, in_channels, tps, ps, ps)` the input *before* the conv.
> When the conv is reinitialised to N≠native channels, `ensure_conv3d_patch_embed` now also updates that
> parent attribute (`_sync_parent_in_channels`); otherwise the pre-conv reshape would split with the wrong
> channel count and raise a shape error at the first forward.

### Encoder-specific `--ct_volume_size` constraints
- **Atlas / Pillar0**: must be large enough to yield the checkpoint's fixed multiscale layout — use
  **≥ `128,128,128`** (grid = D/8, H/8, W/4 must give the checkpoint's scale count). `attach` syncs the
  encoder's expected size to `--ct_volume_size`.
- **Qwen-VL as the 3D encoder**: volume dims must be divisible — `T % temporal_patch_size == 0` and
  `H, W % (patch_size * merge_size) == 0` (Qwen defaults ⇒ `H,W % 28` for 2.5-VL, `% 32` for 3-VL).

### Freezing the 3D encoder
`model.visual_3d` is a normal submodule (`model.visual_3d.encoder` + `model.visual_3d.proj`), so swift's
generic freezing args target it (there is no dedicated `--freeze_vision_3d`; `--freeze_vit` only freezes
the host's 2D tower):
```bash
--freeze_parameters visual_3d                 # freeze the whole 3D backbone (encoder + projector)
--freeze_parameters visual_3d.encoder         # freeze the pretrained 3D encoder, TRAIN the projector
--freeze_parameters visual_3d --trainable_parameters visual_3d.proj   # equivalent to the above
--freeze_parameters_regex '.*visual_3d\.encoder.*'                    # regex form
```

### Debugging
Set `VISION_3D_DEBUG=1` to log, from the live forward pass, each vision encoder's output shape and the
image/video token routing per batch (2D vs 3D, real vs dummy forward). Useful for verifying wiring.

---

## Variable-resolution CT volumes (per-volume token counts)

By default (`--ct_volume_size` set) every volume is resized to that fixed `D,H,W` and expands to exactly
`--vision_3d_max_tokens` tokens. **Omit `--ct_volume_size`** to switch to *variable resolution*: each
volume is smart-resized to a grid-divisible target **near its native shape**, capped so it yields **≤
`--vision_3d_max_tokens`** tokens (so `max_tokens` becomes a per-volume ceiling, not a fixed budget).
Small volumes keep ~native detail; large ones are isotropically downscaled to fit the cap.

- **Qwen-VL 3D encoder only.** Atlas requires a fixed `--ct_volume_size` (its multiscale layout is
  derived from a configured `image_size`); attach raises if you omit it for Atlas.
- **Incompatible with the Perceiver resampler** (it emits a fixed number of latents). Attach raises if
  `--vision_3d_resampler != none` in variable mode. Use `--vision_3d_resampler none`.
- **Works for any Qwen tower, host-matched or not.** The patch geometry (`patch_size / temporal_patch_size
  / merge_size`) used to resize and to count tokens is read from the **`--vision_3d_model`'s own config**
  (threaded onto the template), so it's correct even if the 3D encoder differs from the host (e.g. host
  Qwen3.5 patch=16 + 3D encoder Qwen2.5-VL patch=14). Falls back to the host processor geometry only if
  that config read fails.
- Consider `--padding_free true` / `--packing true`, since sequence lengths now vary across the batch.

Mechanics: the loader (`smart_resize_dims`) picks the target; the CT template computes each volume's real
token count and expands its `<video>` block accordingly (per-volume synthetic `grid_thw`), keeping the
volumes as a **list** (different sizes can't be stacked); the backbone projects the encoder's native
per-volume tokens with **no pooling / no resampler**. Verified: a list of differently-sized volumes
produces exactly `Σ per-volume token counts` embeddings.

## Spatial-Temporal Attention (STA): temporal / z-axis modeling

A Qwen-VL vision tower attends **only within each slice-group** (its `cu_seqlens` are per-frame) and its
RoPE + learned pos-embed are **2D (H,W) only** — so it never mixes information across the z-axis. Passing
`--vision_3d_sta true` injects a lightweight **temporal self-attention** into each Qwen vision block,
between the existing spatial attention and the MLP (own pre-norm `norm3` + residual):

```
h = h + spatial_attn(norm1(h))   # unchanged, per-frame
h = h + temporal_attn(norm3(h))  # NEW: across the gt temporal groups, per spatial location
h = h + mlp(norm2(h))            # unchanged
```

- **Temporal heads** = `round(spatial_heads × 0.25)` (override with `--vision_3d_sta_num_temporal_heads`),
  same `head_dim` as spatial → fewer params.
- **Separate 1D RoPE** over the temporal (`gt`) axis — the principled z-position signal (distinct from
  the 2D spatial RoPE).
- **Zero-initialised output projection** → at step 0 the temporal path contributes exactly 0 and the
  tower reproduces the original bit-for-bit (verified `max|Δ| = 0`). It is **not frozen** — it trains up
  from zero, preserving pretrained spatial modelling initially.
- **Per-volume reshape via `grid_thw`** → correct for a single volume, a batched (concatenated) set, and
  **variable-resolution** volumes (different `gt`). The tower's `grid_thw` is stashed by a forward
  pre-hook and read inside the STA block.
- **Fully opt-in / backwards-compatible.** Off by default; nothing in `sta.py` runs unless enabled. The
  encoder output shape is unchanged, so token counts, splicing, batching, and variable-res are unaffected.
- Atlas is unsupported (it is natively 3D); attach raises if `--vision_3d_sta` is set with a non-Qwen tower.

Cost: adds attention to the already-dominant encoder (cheaper per-head with 4× fewer heads, but real);
pair with the throughput levers (coarser patch, ZeRO-2, etc.). STA params round-trip via `vision_3d_config`
(`sta_enabled`, `sta_num_temporal_heads`) so a reloaded checkpoint rebuilds the identical architecture.

## Batched volume encoding & frozen-2D behavior

- **Batched encoding (Qwen towers).** The Qwen adapter encodes all volumes of a micro-batch in **one**
  tower forward by concatenating their patches with a multi-row `grid_thw` (`cu_seqlens` isolates each
  volume). Verified **bit-identical** to the per-volume loop in fp32 (bf16 differs only by benign
  flash-attention accumulation order). **Windowed towers (Qwen2.5-VL)** reorder tokens across the packed
  sequence, so they are **not** batch-equivalent and fall back to the per-volume loop automatically
  (detected via `window_size`/`fullatt_block_indexes`). Atlas has its own batched adapter.
- **Frozen 2D tower skips the dummy forward.** swift's training path runs a zeroed dummy image through
  `model.visual` on no-image batches purely to keep the 2D tower in the DDP graph. That is now **skipped
  when the 2D tower is frozen** (`--freeze_vit true` + no images), avoiding a wasted per-step forward
  (and, under ZeRO-3, a wasted all-gather of the frozen tower). The dummy is still emitted when the 2D
  tower is *trainable* and a batch has no images (needed for gradient-sync).

## Inference / generation

CT volumes are routed through `model.visual_3d` during **generation** as well as training, but **only on
the `transformers` (PyTorch) inference backend** — `swift infer`/`swift eval`/`swift deploy` with
`--infer_backend pt`. The template's `_post_encode` splices the 3D volume features into the `<video>`
positions once at prefill; the 2D image path is left entirely to native HF (images / X-rays are
encoded and merged exactly as upstream). A request with no CT volume is byte-for-byte the upstream path.

**vLLM / SGLang / LMDeploy are NOT supported for CT volumes.** Those engines re-implement the model
forward and know nothing about `model.visual_3d`, `pixel_values_volumes`, or CT preprocessing — the 3D
encoder is silently bypassed. Use the `pt` backend for CT inference. (2D-only requests still work on any
backend.)

## Checkpoints: saving & reloading a trained 2D+3D model

A full fine-tune (`--tuner_type full`) saves the **whole** model — including `visual_3d.encoder.*` and
`visual_3d.proj.*` — into the checkpoint's safetensors, and records a `vision_3d_config` block in
`config.json` describing how the 3D encoder was built (backbone id, `in_channels`, `encoder_dim`,
`max_tokens`, adapter type, `resampler`, `ct_volume_size`, windows, `variable_tokens`, `sta_enabled` /
`sta_num_temporal_heads`, …) — so a reload rebuilds the identical architecture (resampler, variable mode,
STA temporal blocks) before overlaying the trained weights.

On reload (`--model /path/to/checkpoint`), swift rebuilds the 3D skeleton from that saved config and then
**overlays the trained `visual_3d.*` weights** from the checkpoint (the base VLM load drops them as
unknown keys, so they are loaded explicitly). The attach step fires automatically when `config.json` has
a `vision_3d_config`, and unset `--vision_3d_*` / `--ct_*` flags are backfilled from it — so a reload
script can be as small as `--model <ckpt> --template qwen2_5_vl_ct`. Re-passing the flags still works and
always wins. Note: the base 3D backbone (e.g. Pillar0) is still fetched to rebuild the module skeleton
before the trained weights overlay it, so keep `--vision_3d_trust_remote_code true` reachable on reload.

> **LoRA caveat:** under `--tuner_type lora` the 3D encoder is frozen and PEFT saves only adapters, so
> `visual_3d.*` is not written to the adapter checkpoint. Use `full` (or explicitly make `visual_3d`
> trainable and merge) if you need the trained 3D encoder in the saved checkpoint.

> **torch 2.9 Conv3d:** for torch `2.9.x` ms-swift globally replaces `Conv3d.forward` with an unfold+linear
> fast path (`swift/model/utils.py:_patch_conv3d`) that only models patchify convs (stride==kernel,
> padding==0). The Atlas encoder uses overlapping/padded convs, so that path now **falls back to the
> native Conv3d forward** for any non-patchify geometry instead of raising — required for Atlas to run
> at all (training and inference) on torch 2.9.

---

## Important existing ms-swift arguments (selected)

| Argument | Notes |
|---|---|
| `--model` | Host VLM = LLM + its 2D vision tower. E.g. `Qwen/Qwen2.5-VL-3B-Instruct`. |
| `--tuner_type` | `full` (train all params; the 3D encoder + projector are trainable) or `lora` (only LoRA adapters on the LLM are trained; the 3D encoder is frozen). |
| `--freeze_vit` / `--freeze_aligner` / `--freeze_llm` | Freeze the host's 2D vision tower / aligner(merger) / LLM. `--freeze_vit` defaults **True**; set `false` to also train the 2D encoder. Does **not** affect `visual_3d`. |
| `--learning_rate`, `--vit_lr`, `--aligner_lr` | Base LR, and separate LRs for the 2D vision tower / aligner. |
| `--torch_dtype` | `bfloat16` recommended on H100/L40. (The Atlas 3D encoder always runs in float32 internally for numerical safety; only its projected output is cast to the LLM dtype.) |
| `--attn_impl` | `flash_attention_2` or `sdpa`. **Note:** with transformers 5.12 + flash-attn we hit `flash_attention_forward() got multiple values for 'cu_seq_lens_q'` in the attention path; `sdpa` is a clean fallback. |
| `--use_liger_kernel` | Enable Liger kernels (fused RMSNorm/SwiGLU/cross-entropy) to cut memory/time; compatible with the CT templates. |
| `--deepspeed` | `zero2` / `zero3` for sharded multi-GPU training. |
| `--max_length` | Max total sequence length (text + all vision tokens). CT volume tokens (`--vision_3d_max_tokens` each) count toward this. |
| `--per_device_train_batch_size`, `--gradient_accumulation_steps` | Global batch = micro-batch × data-parallel size × grad-accum. |
| `--dataset`, `--split_dataset_ratio`, `--dataset_shuffle` | Dataset path(s), train/val split, shuffle. |
| `--ddp_find_unused_parameters` | Set `true` if some batches lack a modality so an encoder goes unused under DDP (e.g. CT-only data with `--freeze_vit false`). The 3D encoder always runs a (dummy) forward, so it's DDP-safe by itself. |
| `--eval_steps`, `--save_steps`, `--logging_steps`, `--output_dir` | Standard trainer controls. |

---

## Dataset format
Standard swift chat format; CT volumes go in `videos` and images in `images`:
```json
{"messages": [{"role": "user", "content": "<video>Describe this chest CT."},
              {"role": "assistant", "content": "..."}],
 "videos": ["/path/to/volume.nii.gz"]}
```
`<image>` + `images` for 2D, `<video>` + a volume path for 3D, or both in one sample.

**Multi-array volumes (`.npz` / `.safetensors`).** The loader auto-selects the CT array by key preference
(`ct_data`, `volume`, `image`, `data`, `arr`, `pixel_values`) and never falls back to a `mask`/`seg`/
`label`/`annotation` array. So PE-style files that store the CT under `ct_data` alongside a `mask_data`
load correctly with no renaming; datalake volumes under `image` are unaffected.

## Example / quick test
- Training example: [`examples/train/liger/qwen2_5_vl_3b_atlas_3d_ct_liger.sh`](examples/train/liger/qwen2_5_vl_3b_atlas_3d_ct_liger.sh).
- Smoke test + dummy data: [`test_cases/test_dual_encoder_2d3d.sh`](test_cases/test_dual_encoder_2d3d.sh),
  [`test_cases/make_dummy_mixed_data.py`](test_cases/make_dummy_mixed_data.py).
- Model-loading / template / dataset unit tests: [`test_cases/`](test_cases/).
