# Does `--packing true` work with the slow-fast CT architecture?

**Verdict: NO for slow-fast — it does not crash, it SILENTLY DISABLES the slow branch** (degraded model, no
warning). The fast/`<video>` path and the LLM self-attention are packing-correct; only the slow-fast
cross-attention breaks. Investigated read-only; confirmed by `tests/vision_3d/test_packing_slowfast.py` (3/3).

## How packing works here (so the breakage is clear)
- `swift/dataset/packing.py`: `PackingDataset` bin-packs N samples up to `max_length`; it sets
  `template.packing = True; template.padding_free = True`.
- `swift/template/base.py:1898-1909`: padding-free collapses the whole pack into **ONE sequence → `B = 1`**
  (`batch[:] = [self.packing_row(batch)]; assert len(batch) == 1`).
- Cross-boundary attention is prevented by **per-sample `position_ids` resets** + flash-attn **varlen**
  (base.py:742-743); the explicit block-diagonal 4-D mask exists only on the Megatron branch.
- `swift/arguments/sft_args.py:184-195`: packing ⇒ padding_free and **requires `--attn_impl flash_attn`**
  (hard `ValueError` otherwise).

## What is packing-SAFE
- **Multimodal carry-through**: no guard disables packing for volumes; `Volume3DTemplateMixin._data_collator_mm_data`
  (volume_mixin.py:141-159) concatenates every packed sample's volumes **in `<video>` order**.
- **Vision splice**: `_splice_volume_embeds` (volume_mixin.py:349-376) is **count-based, not row-based** — it
  encodes all volumes and `masked_scatter`s them into every `<video>` slot across the B=1 sequence in order. So
  multiple volumes + multiple `<video>` spans in one packed sequence splice correctly. Fast tokens are fine.

## What BREAKS (the slow branch)
1. **Silent disable (the normal case).** `_stash_slow_tokens` (volume_mixin.py:311-347) keys the slow K/V to
   batch rows: `B = input_ids.shape[0]` and the guard `if not slow or len(slow) != B:` (line 321). Packing forces
   `B = 1` while a pack normally holds N≥2 volumes → `len(slow) = N != 1` → guard trips → it installs an
   **all-masked `(B,1)` dummy** (training) or `None` (inference) → cross-attn contributes exactly 0. The slow
   branch is dropped for every packed sequence with >1 volume, with no error (only a debug log if `_ct_debug`).
   → Training proceeds as if slow-fast were OFF. *(test: `test_packing_silently_disables_slowfast`.)*
2. **Leakage (latent edge case).** `LLMCrossAttnBlock.forward` (slowfast.py:120-151) masks only the **key/T**
   axis (`slow_mask`), with **no per-packed-sample query segmentation**. If a pack ever held exactly one volume
   (`len(slow) == 1 == B`), the guard passes and every text query in the packed sequence — including other
   samples' tokens — attends that one volume's slow tokens → cross-sample leakage.
   *(test: `test_crossattn_has_no_packed_sample_isolation`.)*

Contrast (unpacked, 1:1): `B == N` → real padded slow K/V is built and slow-fast runs normally
(*test: `test_unpacked_1to1_builds_real_slow_kv`*).

## Decisive code
- `swift/template/templates/volume_mixin.py:320-336` — `B = input_ids.shape[0]`; `len(slow) != B` guard → disables.
- `swift/template/base.py:1898-1909` — padding-free collapses the pack to `B = 1` (breaks the B↔volume 1:1 assumption).
- `swift/model/vision_3d/slowfast.py:120-151` — cross-attn masks only the key axis (no per-sample block mask).
- `swift/arguments/sft_args.py:184-195` — packing ⇒ flash-attn required.

## Recommendation
- **Do NOT use `--packing true` with slow-fast as-is** — you'd train a fast-only model while believing it's
  slow-fast. (It would be fine for a *fast-only* / no-slow-branch config.)
- If packing's throughput is wanted WITH slow-fast, it needs **segment-aware** support:
  1. In `_stash_slow_tokens`: when packed (B=1, N volumes), build slow K/V for all N volumes and record each
     volume's **query range** (from the packed `position_ids` resets / sample lengths).
  2. In `LLMCrossAttnBlock`: build a **block-diagonal `attn_mask`** so each query range attends ONLY its own
     volume's slow tokens (preventing leakage), instead of the current full-sequence attend.
  This is a real feature (not a flag flip) and should carry its own tests before use.
- Note: packing targets *data/efficiency*, and this run is **compute-bound** (see TRAINING_SPEED_OPTIMIZATION.md),
  so packing's main value (less padding waste) is a throughput/efficiency win, not a data-starvation fix.
