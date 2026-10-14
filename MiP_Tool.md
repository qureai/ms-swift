# MiP Tool — online, model-invoked Maximum-Intensity-Projection tool for the CT VLM

**Status:** design / roadmap. No code written yet.
**Written for:** engineers on the CT-VLM (ms-swift fork) project.
**Author:** drafted with Claude Code, 2026-10-12 (revised to the *external online tool* paradigm; supersedes the earlier "always-attach a baked MiP image" draft).

## 1. Idea in one line

MiP is an **external tool the model calls during generation, online** — *not* a self-generated ("implicit") text tool, and *not* a MiP image baked statically into every sample. Mid-report, when the model decides it needs a sharper look at small high-HU structures, it **emits a MiP tool call with arguments**; generation pauses; an **executor renders the projection from the actual CT volume**; the rendered 2D image is **injected back into the context as an `<image>`**; generation resumes and the model conditions its findings on the projection it just requested.

**How this differs from everything else we built:**

| | Implicit tools (CLS / CHAR / CAC / …) | **MiP tool** |
|---|---|---|
| Who produces the "output" | the model **generates it as text** | **real code runs** and returns a **new image** |
| Execution | none (pure SFT text) | **external executor at run time (online)** |
| What the model then attends to | its own tokens | **pixels it did not generate** |
| Turn structure | single assistant turn | **multi-turn**: call → tool-image → continue |

The implicit tools teach the model to *say* structured context; the MiP tool lets it *fetch new visual evidence*.

## 2. Why — the motivation

The 3D tower pools the volume heavily (fast tokens = a `16×8×8` grid; slow tokens capped; encoder `--max_tokens`). That pooling **under-resolves small, high-HU structures** — the model's measured weak spots:

- **Pulmonary nodules** — generic `pulmonary_nodule` AUROC ~0.59 (weakest major tag); 3D nodule-bbox grounding ≈ 0 IoU at ckpt-7200.
- **Coronary / vascular calcium** — `coronary_artery_calcification` AUROC ~0.66; cac Agatston magnitude never converges.
- **Vessels / emboli** — small tubular high-contrast structures.

MiP is the standard radiology maneuver for exactly these: collapse the brightest voxel along a ray into a crisp, full-in-plane-resolution 2D image so a 3 mm nodule or a calcified vessel "pops." The base 2D encoder (`model.visual`, high effective resolution) resolves that detail far better than the pooled 3D tokens. **minIP** (minimum projection) is the dual for **low**-HU findings (emphysema, air-trapping, airways). Making it an *online tool* (rather than an always-on extra image) means the cost is paid **only when the model asks**, and the model learns **when** a projection actually helps.

## 3. What a MiP is → the tool's arguments

From an HU volume `(D=Z, H=Y, W=X)` (`read_hu_volume`), a single projection is fully specified by four knobs, which become the **tool-call arguments**:

- **`projection`**: `MIP` (max, bright findings) · `MinIP` (min, dark findings) · `AIP` (avg).
- **`plane`**: `axial` (max over Z→`H,W`) · `coronal` (over Y→`D,W`) · `sagittal` (over X→`D,H`).
- **`slab_mm`**: thin-slab (10–30 mm) keeps nodules legible; full-volume is cluttered for a thick chest. Convert mm→#slices via `slice_thickness`/`spacing_between_slices` (both in the volume master).
- **`region`**: *where* to project — one of `full` · an anatomy label (`right_lower_lobe`, resolved to a z-range/box via the omnict anatomy boxes) · an explicit **z-xy bbox `[z0,z1,x0,y0,x1,y1]`** (our existing normalized convention).
- **`window`** (optional): `lung` (nodules/parenchyma) · `calcium`/`mediastinal` (vessels/calcium) — applied before projecting (`apply_hu_windowing` + `ANATOMICAL_WINDOWS`).

**v1:** constrain to a **small menu** (e.g. `coronal/axial × {MIP,MinIP} × {lung,calcium} × full-or-lobe`) rather than free-form, so both the executor and the learned action space stay bounded.

## 4. Tool-call protocol

- **Call surface:** the model emits `<MIP_CALL>{json args}</MIP_CALL>` (register these as special tokens, same way as `tools_tokens.txt`). `</MIP_CALL>` is the **generation stop trigger**. (A JSON/function-call schema is equivalent; the point is a parseable, stoppable boundary.)
- **Tool response turn:** an **`<image>`** holding the rendered MiP (plus optional one-line text naming the view). The image enters through the **existing 2D `<image>` path** — see §6.
- **Multi-turn shape** (identical at SFT and inference):

```
human: Generate the radiology report for this CT volume.\n<video>
gpt:   <CLS_TOOL><MULTIPLE_PULMONARY_NODULES> ...</CLS_TOOL>
       <MIP_CALL>{"projection":"MIP","plane":"coronal","slab_mm":10,"window":"lung","region":"right_lower_lobe"}</MIP_CALL>
tool:  <image>                      # ← executor renders & injects the real MiP here
gpt:   <NODULE_DETAIL_TOOL>query: NODULE\n3 nodules, largest 6 mm, RLL ...</NODULE_DETAIL_TOOL>
       <ANSWER>{ ...mini report... }</ANSWER>
```

The `<video>` (3D volume) stays on turn 1; the MiP is always an **`<image>`** (hard rule, §6.3).

## 5. The online loop (serving & RL rollout)

The executor runs live — this is the "online, not offline" requirement:

1. Generate until the `</MIP_CALL>` stop sequence.
2. Parse args; resolve `region` (anatomy label → box via the series' anatomy map, or use the bbox as-is).
3. **Execute:** `ct_mip.make_mip(read_hu_volume(volume_uid), **args)` → windowed 8-bit RGB PIL. The volume is the same uid/path carried on the turn-1 `<video>`.
4. Wrap as a `tool`/image turn and **resume generation** with the appended context.
5. Repeat until `<ANSWER>` or a **hard cap** on MiP calls per generation (e.g. ≤2) to bound latency and prevent loops.

Engineering choices:
- **Resume strategy:** re-encode the growing message list for v1 (simplest, correct); optimize to KV-cache continuation + incremental image splice later.
- **Caching:** memoize by `(volume_uid, canonicalized args)` so repeated/near-identical calls don't re-render.
- **Parity:** the *same* executor must run in the RL/data rollout and at inference, or train/infer behavior diverges.

## 6. How the returned image is consumed (architecture — mostly additive)

The fork already lets `<image>` and `<video>` coexist, so the injected MiP needs **no new encoder and no new splice/mask code**.

- **2D path (what MiP uses):** a normal `<image>` → processor image_processor → expanded to `grid.prod()//merge²` ids → encoded by `model.visual` → `masked_scatter`-ed into `inputs_embeds` on `image_token_id` (`swift/template/base.py:2285-2338`, scatter `:2331`). CT template image block: `qwen.py:1417-1427`.
- **3D path (unchanged):** the volume rides `<video>` → `visual_3d` slow-fast (`volume_mixin.py`, `backbone.py`, `slowfast.py`). The slow-token cross-attn hooks read their own K/V buffers and are **untouched** by adding an image.
- **MiP generation reuses existing IO:** `read_hu_volume` (`ct_volume_io.py:76`) → `(D,H,W)` HU float32; `apply_hu_windowing` + `ANATOMICAL_WINDOWS` (`ct_windowing.py:17,33`) window+scale. `make_mip` = read → (optional slab/region crop) → project along axis → window → 8-bit → RGB PIL.

### 6.3 The hard rules
- **MiP must be an `<image>`, never a `<video>`** — the CT template rejects a non-volume `<video>` (`qwen.py:1431-1434`).
- **Token-count accounting must stay exact** — each `<image>` placeholder count must equal `model.visual`'s emitted token count or `masked_scatter` size-mismatches.
- **One volume per sample** stays true (slow-fast assumes it, `volume_mixin.py:321`); MiP adds *images*, not volumes.

## 7. Teaching *when / where* to call (the decision policy)

Because the tool is model-invoked, the "when" is **learned**, supervised two ways:

1. **SFT on multi-turn tool traces.** Build traces where a trigger finding → the matching call → **a really-rendered MiP** (executor runs during data prep / in a tool-executing dataloader) → improved downstream findings. Triggers:
   - MIP: `NODULE`, `MULTIPLE_PULMONARY_NODULES`, `GROUND_GLASS_NODULE`, `TREE_IN_BUD_OPACITY`, `PULMONARY_EMBOLISM`, metastasis, miliary.
   - MinIP: `EMPHYSEMA`, `BRONCHIECTASIS`, `AIR_TRAPPING`, `MOSAIC_PERFUSION`, `AIRWAY_NARROWING`.
   - **Gold positives:** reports that literally say *"best appreciated on MIP / maximum intensity projection / MinIP"* — the radiologist used a projection, so should the model.
   - **Negatives:** clean / non-nodular-non-airway studies → **no call** (teach restraint; same spirit as our 25% negatives).
2. **RL refinement (`swift_rl`).** External tool + verifiable reward is a natural fit: reward calling MiP when it **improves** the downstream metric (nodule count/IoU, cac magnitude), penalize useless calls (token/latency cost). Likely **SFT-then-RL**.

**Where** (the `region` arg) is resolved from the triggering finding's location (`complete_cls` lobe → anatomy box), or from GT nodule boxes (segmed / nodule_fpr); default = whole-lung coronal slab.

## 8. Phased plan

- **P0 — `ct_mip.py` util (pure, testable):** `make_mip(hu_volume, projection, plane, slab_mm, region, window) -> PIL`. Unit-test shape/orientation (synthetic bright blob projects to the right 2D spot). Lowest risk, do first.
- **P1 — executor + call parser:** `</MIP_CALL>` stop-sequence detection, arg parse/validate, `region` resolution, render-on-call given a volume uid. Standalone (no generation loop yet).
- **P2 — online generation loop (swift infer):** generate → stop → parse → render → inject `<image>` turn → resume; call cap + cache. This is the heart of "online."
- **P3 — tokens + smoke:** register `<MIP_CALL>`/`</MIP_CALL>` (+ any arg tokens); one interleaved call end-to-end — image lands via 2D path, counts exact, forward + rope OK with both grids, no-call path unchanged.
- **P4 — SFT tool-trace data + warm-start:** generate multi-turn traces (triggers + report MIP-mentions) with the executor producing *real* MiP responses; short warm-start from a current checkpoint.
- **P5 — RL + eval/ablation:** RL to tune when-to-call; eval **MiP-on vs MiP-off** on the same checkpoint lineage. Go/no-go = measurable lift on `pulmonary_nodule`, `coronary_artery_calcification`, nodule-bbox IoU, with no regression and acceptable call rate/latency.

## 9. Design decisions / open questions

1. **Frozen vs trainable 2D encoder.** Current runs `--freeze_vit true` → `model.visual` frozen; a frozen tower sees MiP (unusual grayscale projection) out-of-distribution. Options: (a) rely on the LLM to read frozen-encoder MiP tokens; (b) small trainable **MiP adapter** after `model.visual` (most contained, keeps checkpoint self-describing); (c) unfreeze last few `visual` layers. Pick before P4.
2. **Resume cost:** re-encode (simple) vs KV-continuation + incremental splice (fast). Start re-encode; the call cap keeps it affordable.
3. **Action space:** fixed menu (recommended v1) vs free-form args. Fixed menu also simplifies RL.
4. **Loop safety:** hard cap on calls/generation; dedupe identical calls; timeout on the executor.
5. **vLLM gap:** the vLLM port does **not** handle volumes yet, so MiP-in-vLLM is blocked until that lands; v1 serving is swift's own (py) infer loop.
6. **Training-time execution:** render live in a tool-executing dataloader/rollout (true online) with an optional disk cache keyed by `(volume_uid, args)` — do **not** pre-bake a static MiP dataset (that's the offline design we're explicitly moving away from).
7. **Orientation / coords:** `read_hu_volume` is `(Z,Y,X)`; map the projection axis to the intended radiological view; if MiP is ever used to emit 2D boxes, define its coordinate convention explicitly.
8. **Reward shaping (RL):** metric-improvement minus a per-call token/latency penalty; avoid rewarding calls that don't change the answer.

## 10. Risks / blockers

- **Does it actually help?** The bet is that an on-demand high-res 2D projection rescues the small-structure tasks the 3D tower under-resolves — P5 ablation is the go/no-go.
- **Online-loop latency:** render + re-encode per call; mitigated by call cap + cache, but it's real added inference cost.
- **Frozen-encoder domain gap** (§9.1) — measure in P3/P5; may force an adapter.
- **Train/infer parity** — the executor must be byte-identical across rollout and serving.
- **`<image>` not `<video>`; exact token accounting; one volume/sample** (§6.3) — same failure classes that bit slow-fast `<video>` expansion.
- **vLLM volume gap** — no volume support in the port yet → MiP serving limited to swift infer for now.

## 11. Key file references (for implementers)

- 3D volume path (unchanged): `swift/template/templates/qwen.py:1379,1406-1456`; `swift/template/templates/volume_mixin.py:97,126,230-278,311-376`; `swift/model/vision_3d/backbone.py:27,106,159-193`; `swift/model/vision_3d/slowfast.py:22-60,112-213`.
- 2D image path / token merge (how the MiP response is consumed): `swift/template/base.py:2285-2338` (scatter `:2331`); CT image block `qwen.py:1417-1427`; rope `qwen.py:471-489`.
- CT IO + windowing (MiP executor source): `swift/utils/ct_volume_io.py:76-116,165-212`; `swift/utils/ct_windowing.py:17-89`.
- Config/args (register tool + tokens): `swift/arguments/base_args/template_args.py:147-161,208-255`; attach + `vision_3d_config` persistence `swift/model/vision_3d/attach.py:356-384`.
- Inference loop to extend for the online tool: swift infer entrypoints (py path); vLLM port is out of scope until volumes are supported there.
