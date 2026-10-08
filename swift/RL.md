# Radiology VLM — Reinforcement Learning (GRPO/GSPO)

End-to-end documentation of the RL pipeline for the 2D chest-X-ray VLM: the environment and
from-source vLLM build, the architecture changes made to ms-swift, the reward design, the curriculum
and safety-score data construction (with counts), and the training hyperparameters.

RL uses **GSPO** = GRPO with sequence-level importance sampling (`importance_sampling_level: sequence`).
The policy is a Qwen3.5 VLM; rollouts are generated with a colocated vLLM engine and scored by a
composite, task-routed radiology reward. All work lives in the `swift_infer` conda env; **`swift_july`
was left untouched.**

---

## 1. Environment & vLLM build

The stock `swift_infer` had no runnable vLLM (all transformers-5-capable wheels are compiled for CUDA
13, but every node runs driver 535 → CUDA 12 only). vLLM was therefore **built from source for
CUDA 12.8** so it links `libcudart.so.12` and runs on the 535 driver.

| Component | Version / note |
|---|---|
| vLLM | `0.26.0+cu128`, **built from source** (`git+https://github.com/vllm-project/vllm.git@v0.26.0`) |
| torch | `2.11.0+cu128` |
| transformers | `5.9.0` (needed for swift's `transformers.trainer.create_repo`; 5.14 removed it) |
| deepspeed | `0.19.5` |
| torchcodec | `0.11.1+cu128` (the pulled build was cu13 → replaced) + conda-forge `ffmpeg 7` |
| **sentence-transformers** | **`6.1.0`** — REQUIRED by the embedding sub-reward + `A_attr`; install `--no-deps` to protect transformers/torch/vllm |
| flashinfer | **removed** — its runtime JIT of `sampling.cu` fails here; vLLM falls back to its own compiled flash-attn + native sampler |
| cmake (build only) | pinned `<4` (3.31.x) — CMake 4.x breaks the conda-nvcc CUDA compiler-id detection |

> **Critical — embedding reward dependency.** The `radiology` reward's embedding term (and the
> `A_attr` attribute agreement) load the encoder via `SentenceTransformer`. If **`sentence-transformers`
> is not installed**, `EmbeddingScorer.cosine_pairs` catches the `ModuleNotFoundError` and returns `[]`,
> so those terms are **silently dropped** and the objective quietly changes (this is what happened in
> the early smokes — rewards were non-zero purely from the verifiable terms while the "main" embedding
> signal was off). Install it and verify before any real run:
> ```
> pip install --no-deps sentence-transformers      # in the swift_infer env
> python -c "import sentence_transformers; print('ST', sentence_transformers.__version__)"
> ```
> The reward now also logs a loud `EmbeddingScorer: sentence_transformers is NOT importable ...` error
> at construction if it is missing. In the first real run, confirm the `embedding` / `attr` reward
> components are non-zero.

**Runtime env vars** (set in the launch sbatch):

```
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_FLASHINFER_SAMPLER=0
export LD_LIBRARY_PATH=$ENV/lib:$ENV/targets/x86_64-linux/lib:$LD_LIBRARY_PATH
export NCCL_P2P_DISABLE=1                 # L40S have no NVLink
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MAX_PIXELS=1003520                 # 2D X-ray image budget (Qwen3.5 patch16)
```

Build/verify scripts live under `/cache/fast_data_nas71/janhavi/data/rl_training/smoke/`
(`vllm_build_cu128.sbatch`, `vllm_verify.sbatch`). After building vLLM, run
**`/cache/fast_data_nas71/janhavi/data/rl_training/setup_swift_infer_env.sh`** — the single
reproducible runtime setup: it pins transformers 5.9.0 + deepspeed, installs the cu128 torchcodec +
ffmpeg, removes flashinfer, installs **sentence-transformers** (embedding-reward requirement), and
reinstalls swift editable so the `vllm.general_plugins` (`swift_vision_3d`) entry point is active for
spawned EngineCore workers. It ends with a verification block (asserts sentence_transformers imports
and the plugin is registered).

---

## 2. Architecture changes (ms-swift)

### 2.1 `visual_3d` (secondary CT tower) support in the vLLM rollout engine

The CT SFT checkpoints carry a Qure-custom secondary 3D vision tower (`visual_3d.*`, 445 tensors)
attached by swift's HF path; vanilla vLLM has no module for it and its loader raises. A drop-in
subclass was added so the rollout engine builds and loads the 3D tower and mirrors the 2D flow.

- **New:** `swift/model/vision_3d/vllm_qwen3_5.py`
  - `Qwen3_5With3DForConditionalGeneration(Qwen3_5ForConditionalGeneration)` — builds `visual_3d`
    from the checkpoint's `vision_3d_config` (rebuilt from the host `vision_config` via
    `Qwen3_5VisionModel` + Conv3d-inflated patch embed + STA — **no 8 GB source download**), then
    **direct-installs** the 445 trained tensors by `.data` assignment (robust to empty/meta/
    ZeRO-3-partitioned params). Adds a `volume` modality that mirrors `_process_video_input`
    (patchified `pixel_values_volumes` + `volume_grid_thw` → `visual_3d` → per-volume split → merged
    into `<video>` positions).
  - **Backwards compatible:** self-gates on `vision_3d_config` — a plain 2D checkpoint builds nothing
    3D and behaves exactly like the vanilla class.
  - **Registration:** conditional in-process hook `VllmEngine._maybe_register_vision_3d()` (colocate
    is in-process, so this suffices for RL) **plus** a `vllm.general_plugins` entry point in
    `setup.py` (`swift_vision_3d`) so it also survives vLLM's spawned `EngineCore` workers
    (`swift infer` / `swift deploy`).
- **Edited:** `swift/model/vision_3d/attach.py` — the policy-side `visual_3d` weight load now uses
  `deepspeed.zero.GatheredParameters` so ZeRO-3-partitioned params gather → load → re-partition
  (a plain `load_state_dict` fails with empty `[0]` shards under ZeRO-3).
- **Edited:** `swift/infer_engine/vllm_engine.py` — added `_maybe_register_vision_3d()` before engine
  construction.
- The transient site-packages patch to vLLM's `qwen3_5.py` was **reverted** — upstream vLLM stays
  pristine; the custom subclass owns all 3D handling.

### 2.2 swift ↔ vLLM 0.26 compatibility fix

- **Edited:** `swift/rlhf_trainers/utils.py` — `finish_vllm_weight_reload` guards
  `isinstance(module, FusedMoE)` with `isinstance(FusedMoE, type)`. In vLLM 0.26 (Triton disabled)
  `FusedMoE` resolves to a non-type stub, which crashed the first weight sync on our dense model.

### 2.3 Template fix (image patch alignment)

The 4B CT checkpoint's `args.json` recorded `template: qwen2_5_vl_ct` (Qwen2.5-VL, `version='v2_5'`,
patch 14 / factor 28) for a **Qwen3.5** model (patch 16 / factor 32). The v2.5 template did not pass
`image_patch_size=16` to `qwen_vl_utils.fetch_image`, so rollout images were resized to 28-multiples
(e.g. 840 px wide) that are **not** divisible by the Qwen3.5 patch size → patchify error. **Fix:**
override `template: qwen3_5` in the training config (v3, patch 16 → images align to 32). `visual_3d`
still attaches because it is driven by `vision_3d_config`, not the template.

> When the Qwen3.5-2B SFT checkpoint arrives, verify its `args.json` `template`; keep the
> `template: qwen3_5` override if it again records a v2.5 template.

---

## 3. Reward design

A single registered ORM, **`radiology`** (`examples/train/grpo/plugin/radiology_reward.py`), routes
per row on the dataset `task` column. Every sub-reward returns a value in **[0, 1]**; each task's
score is a **renormalized convex combination** of its active sub-rewards (so the total stays in
[0, 1] and dropping a term reallocates its weight).

### 3.1 Task router

| `task` value | sub-reward | notes |
|---|---|---|
| `ontology`, `cls_json`, `cls_tokens` (default) | **Report reward** | main embedding + cls path |
| `asp` | **ASP reward** | symbolic atoms + section embedding |
| `kgraph` / `kg` | **Knowledge-graph reward** | triple matching |
| `comparison` | **Comparison reward** | shared/unique + CoT/summary embedding |
| `bbox` | **BBox reward** | DIoU localization (verifiable) |
| compound (`"a,b"`) | segmented per sub-task, each scored by its router entry |

GT source of truth is the pre-extracted **`reference`** column (normalized `finding_set`,
`prominence`, `absent_set`, section texts, triples); raw `solution` text is a fallback.

### 3.2 Report reward — outer mixing weights

`w_* need not sum to 1; renormalized over active terms.`

| term | weight | kind |
|---|---|---|
| embedding (`w_emb`) | **0.35** (base) | embedding — the main signal |
| cls (`w_cls`) | 0.25 | verifiable (presence) |
| bbox (`w_bbox`) | 0.15 | verifiable |
| token (`w_token`) | 0.10 | verifiable |
| consistency (`w_consistency`) | 0.10 | format/logic |
| format (`w_format`) | 0.03 | format |
| length (`w_length`) | 0.02 | format |

- **CLS internals:** sensitivity 0.40 (recall, FN-heavy), specificity 0.20 (→ Youden), asymmetric
  Tversky overlap 0.40 (`tversky_fn=1.5 > tversky_fp=1.0` → FN penalized more than FP). Presence is
  **prominence-weighted**.
- **Token internals:** cls_tokens/cls_json weight each finding by prominence
  (`1 + token_gain·prom/5`, `token_gain=2.0`); other tasks use plain word F1.
- **Embedding internals:** cosine → `(cos+1)/2`; convex over sections findings 0.45 / impression 0.35
  / reasoning 0.20; encoded by the fine-tuned radiology encoder (`--reward_embedding_model`).

**Dynamic embedding weight (two multiplicative modulators):**

```
w_emb_eff = w_emb · (1 + emb_abn_gain · min(n_abn, emb_abn_cap)/emb_abn_cap) · safety_score
            └───── base ─────┘  └──── abnormality-count scaling ────┘        └─ GT reliability ─┘
```
`emb_abn_gain=1.0`, `emb_abn_cap=6` (doubles at ≥6 GT abnormalities). `safety_score ∈ [0,1]` is the
per-row GT-completeness weight (§4.3). Because the combination is renormalized, shrinking
`w_emb_eff` automatically hands weight to the verifiable terms (cls/FP, bbox, token).

### 3.3 ASP reward

Weights: `sections=0.40`, `progression=0.15`, `finding=0.20`, `absent=0.15`, `structure=0.10`.
- **Embedding terms** (scaled by `safety_score`): `sections` (per-section cosine vs GT
  scene_graph/explanation/cot/inference), `progression` (within-rollout consecutive-section cosine).
- **Verifiable terms** (unscaled, dominate when safety is low): `finding` (Tversky of `finding()`
  atoms vs GT `finding_set`), `absent` (Tversky of `absent()` atoms — real specificity), `structure`
  (parseable symbolic graph).

**ASP block delimiter.** The model tokenizer has `|<ASP>|` (id 248077) and `|</ASP>|` (id 248078) as
**single special tokens** (`examples/train/liger/token.txt`), so a trained model emits those at the
ASP block boundary — note this is *not* `<|ASP|>` (which the tokenizer splits into 5 tokens, and which
only appears as a legacy string in older gemini SFT data). `_asp_body()` / `_ASP_OPEN_RE` /
`_ASP_CLOSE_RE` match **both** the special-token form and the legacy string, and `segment_completion`
slices the ASP block by these tokens **before** the generic `<task>…</task>` delimiter loop (whose
regex would otherwise match `<ASP>` inside `|<ASP>|`). This lets a consolidated `asp,ontology` rollout
have its ASP part separated cleanly from the report; standalone `asp` is unaffected.

### 3.4 Knowledge-graph reward

Weights: `triple=0.45`, `holistic=0.25`, `finding=0.20`, `structure=0.10`. Triples matched by
embedding similarity (soft precision/recall); holistic graph similarity; finding-set overlap;
structural well-formedness. *(Not yet safety-modulated — see §7.)*

### 3.5 Comparison reward

Weights: `cov=0.30`, `diff=0.25`, `cot=0.25`, `summary=0.20`. Operates on study **pairs**
(`set1/set2/shared/unique1/unique2/cot_text/summary_text`): coverage + differential correctness
(verifiable, derived from tag sets) and CoT/summary cosine (embedding). *(Not safety-modulated —
comparison images are not in the single-image tag JSON; see §7.)*

### 3.6 BBox reward (fully verifiable)

Weights: `loc=0.45` (mean DIoU of predicted boxes vs same-label GT), `align=0.35` (F-beta of box
labels, `beta=2.0` recall-favored), `degen=0.1` (non-degeneracy), `format=0.1` (parseability). Uses
**DIoU** (informative even at zero overlap), not IoU. No embedding term.

- **`format`** (added after observing bbox runs away into repetition loops): `_format_score(text,
  n_valid)` returns **0** when no box list is parseable at all, multiplies by **0.3** when the output is
  truncated / the list is left unclosed (the repetition-to-max-tokens failure mode), and otherwise
  scores `valid_boxes / parse_attempts` — so a model that emits garbage after valid boxes is penalized.
- **`degen` strengthened**: now counts **any-label** near-duplicate box pairs (previously only
  *different*-label dups were caught, which let a model spam many copies of the *same* label for free),
  plus an **excess-box penalty** `(len(pred) − box_cap)` once `len(pred) > box_cap` (`box_cap=18`,
  `dup_scan=60`). Validated: repetition-loop / truncated / garbage → `format=0`, reward floors at
  `0.10`; a 40-duplicate output → `degen=0`.

### 3.7 Constrained-RL additions (experiment 1)

A set of terms layered onto the **report reward** (implemented in `radiology_reward.py`), following the
principle *each signal judges only what it can judge reliably*: **verifiable presence** carries
completeness + anti-hallucination, the symmetric **embedder is demoted** to a reliability-weighted
semantic term, and the model's **token probabilities** are scored for calibration. **All additions are
OFF by default** (weights 0 / `cal_mode='none'` / `beta_k=0`) → the reward is byte-identical to §3.2
unless enabled, and each term **gracefully no-ops** when its inputs are absent (renormalizing the
convex combination). Tuned via `RADREWARD_*` env vars (see table).

**Presence as constraints (fixed-λ or dual-ascent).** Two verifiable scalars are added as weighted
members of the combo:
- `R_w` = **prominence-weighted recall** = `Σ_{g∈TP} w_g / Σ_{g∈G} w_g` (`w_g` from GT `prominence`; == the cls **sensitivity**). Completeness → blocks under-calling.
- `P` = **precision** = `|TP| / |A|` (1.0 when nothing is predicted). Anti-hallucination → blocks invention.

They enter as `λ_R·R_w + λ_P·P`. **What the constraints do:** `R_w` is a *floor on completeness* — its
price `λ_R` buys reward only by recovering (prominence-weighted) GT findings, so raising it directly
fights under-calling; `P` is a *ceiling on invention* — its price `λ_P` is paid only by not
hallucinating, so raising it fights over-calling. Because they are **decoupled** (a hallucination drops
`P` but not `R_w`; an omission drops `R_w` but not `P` — pinned by `test_precision_recall_decoupled`),
the two prices trade off completeness vs. precision independently.

Two ways to set the prices:
- **Fixed-λ scalarization** (default): `λ_R = w_recall`, `λ_P = w_precision` are constants you pick.
- **Dual-ascent (PPO-Lagrangian)** — *now implemented, env-gated via `RADREWARD_DUAL_ASCENT=1`*. The
  prices become **dual variables** updated once per batch toward constraint targets:
  ```
  λ ← clip( λ + η·(target − batch_mean) ,  λ_min,  λ_max )
  ```
  seeded from `w_recall`/`w_precision`. **What happens when λ is switched to dual ascent:** if a batch's
  mean recall sits *below* `recall_target`, `λ_R` **rises** (recall gets more expensive to ignore) and
  keeps rising until the policy meets the floor; once recall is satisfied, `λ_R` **relaxes** back down so
  the other terms regain weight — a self-tuning price instead of a hand-guessed constant, which removes
  the manual λ search and stops a constraint from permanently dominating the convex combo. `λ_P` behaves
  the same against `precision_target`. Clipping to `[λ_min, λ_max]` keeps any single constraint from
  swamping the mix. The running λ's persist on the reward object across batches and are **logged** as
  `reward/<subtask>/{lambda_recall,lambda_precision}`. **We still use GRPO/GSPO** — this is reward
  shaping (Lagrangian penalty) on top, not a different optimizer. **1_oct run uses dual ascent**
  (`recall_target=0.85`, `precision_target=0.75`, `η=0.05`, `λ∈[0,1]`, seeds `λ_R=0.5`, `λ_P=0.2`).

**Token-probability calibration — weighted Brier** (`cal_mode='brier'`, cls_json only). For each
cls-JSON tag, `p_true = P("true")` read from the rollout's token logprobs (`= exp(lp)` if a `true`
token was sampled at that presence slot, else `1−exp(lp)` — that slot is ~always true/false), and
`y ∈ {0,1}` = GT presence:

```
reward_cal = 1 − [ Σ_t v_t (p_true_t − y_t)²  /  Σ_t v_t ] ,   v_t = w if y_t=1 (present) else 1
```

`w` = `cal_true_weight` (default **6**) up-weights the few present findings against the ~95 absent
tags (class-imbalance / FN-heavy). It reads `response_token_ids` + `rollout_logprobs` — which the vLLM
rollout already attaches to each row — so **no swift-core plumbing**; it no-ops if the token data is
missing, the task isn't cls_json, or the boolean-token count doesn't match the parsed tags. (AuROC was
considered and rejected in favour of Brier: denser per-tag gradient, proper scoring rule, defined for
normals.)

**Attribute agreement `A_attr`** (`w_attr>0`). For each **matched present tag**, for each axis
(`size`/`location`/`characteristics`), **only where BOTH prediction and GT have a non-null value**,
the embedder scores `(cos+1)/2`; the mean over comparable pairs is `A_attr`. GT-silent and pred-only
attributes are **neutral** (not rewarded, not penalized) — contained over-specification, no bbox
geometry or dedicated graders. Needs GT attributes in `reference['attributes']` (populated by the
consolidated data build); no-ops otherwise.

**Embedding reliability `β(n)·safety`.** The embedding weight becomes
`w_emb · β(n) · safety`, with `β(n) = n/(n+k)` (`n` = #GT findings, `k = beta_k`): near 0 on sparse
reports (compressed cosine, case-2 artifacts dominate), rising toward 1 on multi-finding reports.
`β(n)` **replaces** the legacy abnormality-count scaling when `beta_k>0`. `safety` is the GT-completeness
score (§4.3). Optional benign-dip cap `S_used = max(S, emb_baseline)` applies only when `P=R=1`
(perfect presence ⇒ any dip is phrasing).

**Config knobs** (env, read in `RewardConfig.__post_init__`):

| env var | field | default | meaning |
|---|---|---|---|
| `RADREWARD_W_RECALL` | `w_recall` | 0 | `λ_R`: fixed price of `R_w`, **or** the dual-ascent seed |
| `RADREWARD_W_PRECISION` | `w_precision` | 0 | `λ_P`: fixed price of `P`, **or** the dual-ascent seed |
| `RADREWARD_DUAL_ASCENT` | `dual_ascent` | 0 | `1` → `λ_R`/`λ_P` become dual variables updated per batch |
| `RADREWARD_RECALL_TARGET` | `recall_target` | 0.85 | recall floor the dual update drives toward |
| `RADREWARD_PRECISION_TARGET` | `precision_target` | 0.85 | precision floor the dual update drives toward |
| `RADREWARD_DUAL_LR` | `dual_lr` | 0.05 | dual step size `η` |
| `RADREWARD_LAMBDA_MIN` / `_MAX` | `lambda_min`/`lambda_max` | 0 / 1 | clip range for the dual λ's |
| `RADREWARD_CAL_MODE` | `cal_mode` | `none` | `none`\|`brier` |
| `RADREWARD_W_CAL` | `w_cal` | 0 | weight of the Brier term |
| `RADREWARD_TRUE_WEIGHT` | `cal_true_weight` | 6 | present-tag weight `w` in the Brier |
| `RADREWARD_W_ATTR` | `w_attr` | 0 | weight of `A_attr` |
| `RADREWARD_BETA_K` | `beta_k` | 0 | `β(n)=n/(n+k)`; 0 disables (β=1) |
| `RADREWARD_EMB_BASELINE` | `emb_baseline` | 0 | benign-dip cap at `P=R=1`; 0 off |
| `RADREWARD_TVERSKY_FN` / `_FP` | `tversky_fn`/`tversky_fp` | — | FN / FP penalties in the cls overlap (raise FN to fight under-calling) |
| `RADREWARD_CLS_W_SENS` / `_W_SPEC` | `cls_w_sens`/`cls_w_spec` | — | sensitivity / specificity split inside the cls term |
| `RADREWARD_COMPOUND_AGG` | — | `mean` | compound-task aggregation: `mean`\|`geomean`\|`min` (geomean/min stop a sub-task being dropped) |

Logged reward components now include `recall`, `precision`, `calibration`, `attr`, `beta`,
`lambda_recall`, `lambda_precision` (dual-ascent multipliers, constant per batch), and `format` on the
bbox subtask (+ the existing `safety`, `cls`, `embedding`, …).

**Per-task assembly (experiment 1):** the consolidated **cls_json→report** task uses the full stack
(R_w, P, Brier, A_attr, β·safety, embedding on the report segment); **cls_tokens→report** gets the same
minus Brier (no true/false tokens) and minus A_attr on axes it doesn't emit; **asp** and **kgraph** keep
their existing verifiable+embedding rewards (`β(n)` on their embedding parts is a follow-up). Comparison
and bbox are excluded from experiment 1.

---

## 4. Data creation

### 4.1 Source task parquets

`/cache/fast_data_nas71/janhavi/data/rl_training/training/` — 8 prompt-only parquets, one per task,
sharing an identical 11-column schema (`messages`, `images`, `task`, `solution`, `reference`,
`image_id`, `source`, `n_findings`, `prominence_max`, `difficulty`, `difficulty_bucket`). Images
resolve to `/cache/fast_data_nas72/homagni/vlm_training_images/raid3/...`.

| parquet | task | rows | easy / medium / hard |
|---|---|---:|---|
| `cls_json.parquet` | cls_json | 148,229 | 49,319 / 49,430 / 49,480 |
| `cls_tokens.parquet` | cls_tokens | 148,229 | 49,319 / 49,430 / 49,480 |
| `ontology.parquet` | ontology | 148,229 | 49,319 / 49,430 / 49,480 |
| `asp.parquet` | asp | 108,347 | 35,581 / 36,426 / 36,340 |
| `comparison.parquet` | comparison | 82,255 | 27,419 / 27,418 / 27,418 |
| `bbox.parquet` | bbox | 67,260 | 22,420 / 22,420 / 22,420 |
| `compound.parquet` | multi-task (`"a,ontology"`) | 20,000 | 5,063 / 10,029 / 4,908 |
| `kgraph.parquet` | kgraph | 10,000 | 3,334 / 3,333 / 3,333 |
| **total** | | **~732,549** | |

### 4.2 Curriculum assignment

- **Difficulty bucket** (`difficulty_bucket ∈ {easy, medium, hard}`) is precomputed per row from
  `n_findings` + subtlety/prominence and split into thirds.
- **Sub-sampling:** RL does not need the full corpus (each prompt yields `num_generations` rollouts).
  `build_curriculum.py` / `build_curriculum_safety.py` sample **800 prompts per tier per task**, then
  **merge all tasks within a tier and shuffle** (so the 8 tasks *interleave*, not task-blocked).
- **Staging = ordering + no shuffle:** the three stage parquets are listed **easy → medium → hard**
  in the config with **`dataset_shuffle: false`**. Verified mechanism: swift sets
  `self.shuffle_dataset = args.dataset_shuffle` and trl's `RepeatSampler(shuffle=self.shuffle_dataset)`
  → sequential consumption. The model therefore sees easy prompts of every task first, then medium,
  then hard.

**Main curriculum** — `/cache/fast_data_nas71/janhavi/data/rl_training/curriculum/`

| stage | rows | mean safety | % safety==1 |
|---|---:|---:|---:|
| `curriculum_easy.parquet` | 6,400 | 0.794 | 39.5% |
| `curriculum_medium.parquet` | 6,400 | 0.784 | 31.0% |
| `curriculum_hard.parquet` | 6,400 | 0.780 | 28.4% |
| **total** | **19,200** | | |

(The elevated %-safety==1 includes bbox+comparison rows that default to 1.0; see §4.3.)

### 4.3 Safety score assignment

**Motivation.** With an embedding reward, a model that emits *more* detail than a sparse GT is
penalized as a false positive. The **safety score** measures how completely the GT specifies each
present finding, so the embedding reward can be trusted proportionally.

**Source.** Tag-extraction JSONL
`/cache/fast_data_nas71/janhavi/data/vlm_training/23_June_2026_corrected_full_tag_extraction_vlm_mimic_chex_segmed.json`
(636,546 reports). Each report is a structured JSON of 7 sections × abnormality tags; each tag has
`presence` + three attributes `{size, location, characteristics}`.

**Formula** (`build_safety_map.py`):

```
for each tag with presence == true:
    completeness = (# non-null of {size, location, characteristics}) / 3
safety_score   = mean(completeness over present tags)      # normal report (no present tags) -> 1.0
```

- `safety_score == 1.0` ⇔ every present finding is **fully attributed** (or the report is normal).
- The score decreases as attributes go missing — exactly "less attributes present for each tag →
  lower safety."
- Joined onto RL rows by `image_id` (tag-JSON `image` `/raid3/…` → strip `/raidN/` → `image_id`).

**Safety distribution over the 636,546 scored reports:**

| bucket | count |
|---|---:|
| `safety == 1.0` | 107,534 (16.9%) — of which 88,592 are *normal* (no findings) |
| `[0.67, 1.00)` | 181,548 |
| `[0.34, 0.67)` | 338,266 (the majority — GT typically partially attributed) |
| `[0.00, 0.34)` | 9,198 |

**Coverage on the RL data** (fraction of rows whose image is in the tag JSON):

| task | coverage | note |
|---|---:|---|
| ontology / cls_json / cls_tokens / compound | ~99.9% | scored |
| asp | ~99.9% | scored |
| kgraph | 60.3% | partially scored |
| **bbox** | **0%** | internal cxr_data images — not in tag JSON |
| **comparison** | **0%** | study *pairs* — not single-image scored |

Uncovered rows **default to `safety_score = 1.0`** → their embedding weight is unchanged. This is
correct: bbox is fully verifiable, and comparison's embedding is on CoT/summary (pair-based), so
neither should be down-weighted by a single-image completeness score.

**How safety is used.** The `safety_score` column is written into every curriculum parquet and read
by the reward as a per-row kwarg (same path as `n_findings`). The reward multiplies the embedding
weight by it (Report `w_emb_eff *= safety`; ASP `sections`/`progression` `*= safety`). High safety →
embedding-led; low safety → verifiable-led. `safety` is logged as a reward component.

### 4.4 High-quality (`safety == 1`) subset

`build_curriculum_safety.py` also emits a **verified-complete-GT** subset — only rows genuinely
scored `safety == 1.0` (excludes the defaulted bbox/comparison), the "most reliable" data:

| stage | rows | composition |
|---|---:|---|
| `curriculum_hq_easy.parquet` | 3,778 | asp/cls_json/cls_tokens/ontology 800 each + compound + kgraph 55 |
| `curriculum_hq_medium.parquet` | 3,276 | asp 628, cls/ontology 800 each, kgraph 49, compound |
| `curriculum_hq_hard.parquet` | 1,501 | cls/ontology 382 each, asp 292, kgraph 20, compound |
| **total** | **8,555** | |

The subset shrinks with difficulty (hard reports rarely have *all* attributes filled for *all*
findings). It can be used as a phase-1 high-trust warmup or a standalone maximally-reliable run.

### 4.5 Data-build scripts

`/cache/fast_data_nas71/janhavi/data/rl_training/`
- `build_curriculum.py` — original difficulty sub-sample (no safety).
- `build_safety_map.py` — parses the tag JSONL → `safety/image_safety.parquet` (`image_id`,
  `safety_score`, `n_present_tags`, `n_full_attr_tags`); prints coverage + distribution.
- `build_curriculum_safety.py` — rebuilds the curriculum **with** `safety_score` + `n_present_tags`
  columns and emits the `curriculum_hq_*` subset. Tunables: `N_PER_TIER_PER_TASK=800`,
  `N_HQ_PER_TIER_PER_TASK=800`.
- `build_exp1_data.py` — the experiment-1 dataset (§4.6).

### 4.6 Experiment-1 dataset (consolidated cls→report + asp→report, standalone bbox + kgraph, safety ≥ 0.8)

`/cache/fast_data_nas71/janhavi/data/rl_training/exp1/exp1_{easy,medium,hard}.parquet`. Built by
`build_exp1_data.py` from the `safety ≥ 0.8` images (abnormal + **~12% normals**), curriculum-ordered
(easy→medium→hard, tasks interleaved within a tier). Five task types:

- **`cls_json,ontology`** and **`cls_tokens,ontology`** — *consolidated cls→report*: one prompt
  ("Classify … Then generate the ontology report …"), scored by segmenting the completion. The
  reference bundles the cls block `{finding_set, absent_set, prominence, attributes}` (attributes =
  per-present-tag `{size, location, characteristics}` parsed from the cls-json full template, for
  `A_attr`) and the `ontology` block `{findings_text, impression_text, reasoning_text, …}` (report
  text for the embedding term — sourced from `ontology.parquet`'s reference, **not** the 12 GB
  conversation JSON, which the RL data already carries pre-parsed).
- **`asp,ontology`** — *consolidated asp→report*: "Generate ASP reasoning … Then generate the ontology
  report …". asp images fully overlap ontology (108k), so the reference is task-keyed `{"asp":
  {finding_set, absent_set, sections, prominence}, "ontology": {…}}`. The ASP block is delimited by the
  `|<ASP>|` / `|</ASP>|` special tokens and sliced out in `segment_completion` (§3.3).
- **`bbox`** — *standalone* grounding (fully-verifiable DIoU/label reward, no embedding term). The bbox
  images are `internal_cxr` and have **zero overlap** with ontology-report images, so bbox **cannot** be
  consolidated with ontology; it is not in the tag JSON either, so `safety_score` defaults to 1.0 (a
  no-op for the verifiable bbox reward).
- **`kgraph`** — standalone positive-findings graph/triples, filtered to `safety ≥ 0.8`.

Counts: easy 6,155 / medium 6,359 / hard 6,518 (~19k prompts). Per tier: cls_json,ontology 1,500 +
cls_tokens,ontology 1,500 + asp,ontology 1,500 + bbox 1,500 + kgraph 155–518 (kgraph is 60%-covered so
smaller). Comparison is excluded. Tunables: `MIN_SAFETY=0.8`, `N_PER_TIER_PER_TASK=1500`, `NORMAL_FRAC=0.12`.

**Standalone cls datasets (1_oct, `build_standalone_cls.py`).** exp1 only had cls *consolidated* with
the report ("classify … **then** generate the ontology report"), so the policy never learned to emit
**only** the tokens / **only** the cls-json on a bare `Classify …` prompt — at eval it drifts to
bbox/report. Two standalone datasets fix this (same recipe: `safety≥0.8`, ~12% normals, 1500/tier
easy→medium→hard, appended **after** the three curriculum stages in the 1_oct yaml so they do not
disturb the difficulty ramp):
- **`cls_tokens_standalone.parquet`** (4,500 rows, `task=cls_tokens`) — prompt *"Classify the
  abnormalities …"*, solution `<TAG> <TAG> …`.
- **`cls_json_standalone.parquet`** (4,500 rows, `task=cls_json`) — prompt *"… in JSON format with
  presence, size, location …"*, solution the cls-json template; `reference['attributes']` is parsed in
  from the solution so `A_attr` has GT.

**1_oct curriculum rebuild (`build_exp1_downsampled.py` → `exp1_{easy,medium,hard}_ds.parquet`).**
The raw 1_oct epoch (28k rows) took ~10 d at ~131 s/step on L40S vs a 5-day wall → only ~47 % covered,
and the standalone cls (appended last) was never reached. Three transforms fix this:
- **Split `asp,ontology`** (the *only* ASP-containing task in exp1) into two standalone tasks sharing the
  image: **`ontology`** (ontology-only prompt, flat ontology-block reference) and **`asp`** (asp-only).
  Rationale: compounding let a long ASP block (the `sections` GT is ~4–6 k tok) eat the completion budget
  and truncate the shorter, more important ontology panel → under geomean the whole compound collapsed.
  **Only `ontology` is added to the curriculum; ASP is dropped** (→ `asp_standalone.parquet`, 4,500 rows,
  **reserved, not in the config**) for now.
- **Reorder**: each standalone-cls tier is merged **into** its difficulty stage (no longer stranded at the
  end) and tasks are shuffled **within** a stage; the easy→medium→hard order across stages is preserved.
- **Downsample**: `FRAC=0.60` of each task per tier (kgraph kept whole — it has only 155/359/518, from a
  10 k-row source vs 148 k for cls/ontology). Per tier: 6 tasks × 900 + kgraph. Totals **easy 5,555 /
  medium 5,759 / hard 5,918 = 17,232 rows → ~4,308 steps → ~3 d L40S / ~1.3 d 4×H100** (fits the wall).

With ASP removed, `max_completion_length` is lowered **3072→2048** (longest remaining real completion
~800 tok = kgraph; 2048 clears all legit output and only clips runaway repetition), and the batch config
reverts to the proven `per_device_bs=2 / ga=8 / spg=16` (bs=1 was only needed to fit 3072). The sbatch
`--mem` is raised **150G→220G** (24_sep was host-RAM OOM-killed at 15 h under zero2_offload).

**24_sep run.** Config `examples/train/grpo/radiology_gspo_exp1_24sep.yaml` + sbatch
`rl_training/rl_exp1_24sep.sbatch`. Uses the **`23_sep_qwen3_4b_cross_axis_v1/checkpoint-900`**
embedder (replacing 18_sep), and switches the constrained-RL reward terms ON via `RADREWARD_*` env
(§3.7): `CAL_MODE=brier W_CAL=0.15 TRUE_WEIGHT=6 W_RECALL=0.3 W_PRECISION=0.3 W_ATTR=0.1 BETA_K=4
W_CLS=0.1`. scron-safe (`-c 24`, `--mem=150G`); the sbatch runs on any free 8×H100 node and does
`exit $rc` so a crash shows as FAILED (not a misleading COMPLETED from a trailing echo).

**H100 requires `VLLM_FORCE_FA2=1`.** The from-source cu128 vLLM's FlashAttention-3 (Hopper/SM90)
kernel ships PTX this cluster's H100 driver rejects (`unsupported toolchain`), crashing rollout at
model-load. The sbatch sets `VLLM_FORCE_FA2=1`, honored by a local patch in vLLM's
`v1/attention/backends/fa_utils.py` (`get_flash_attn_version`) that downgrades FA3→FA2 (the same
kernel used on L40S). See memory `h100-fa3-ptx-force-fa2`.

**Reward logging → wandb.** `report_to: [tensorboard, wandb]`, project `cxr-rl-grpo`, run
`exp1_24sep_gspo_cxr` (creds from `~/.netrc`; needs `wandb` in the env — installed `--no-deps`). The
`RadiologyRouter` logs, best-effort on rank-0 (`commit=False`), `reward/overall_mean`,
`reward/task_reward/<task>` (per task-string mean), and every sub-reward component as
`reward/<subtask>/<component>` — e.g. `reward/cls_json/{recall,precision,calibration,attr,beta,safety,
cls,token,embedding,…}`, `reward/asp/{sections,progression,finding,absent,structure}`,
`reward/bbox/{loc,align,degen}`, `reward/kgraph/{triple,holistic,finding,structure}`. Smoke first with
`radiology_gspo_exp1_smoke.yaml` + `rl_exp1_smoke.sbatch` (2×H100, `max_steps=4`).

---

## 5. Training configuration & hyperparameters

Config: `examples/train/grpo/radiology_gspo_curriculum.yaml`. Launch (not auto-submitted):
`/cache/fast_data_nas71/janhavi/data/rl_training/rl_curriculum.sbatch` (submit from qure5).

| group | setting | value |
|---|---|---|
| model | `model` | `Qwen3.5-2B_stage6b_alltools/v0-20260906-173237/checkpoint-608` (plain 2D Qwen3.5, no `vision_3d_config`) |
| | `torch_dtype` / `tuner_type` | bf16 / full |
| | `template` / `attn_impl` | `qwen3_5` / `sdpa` |
| reward | `reward_funcs` | `radiology` (router) |
| | `reward_embedding_model` | `.../15_sep_qwen3_4b_contrast_axes_v1/checkpoints/checkpoint-2500` |
| | `scale_rewards` | `group` (group-standardized advantages) |
| data | `dataset` | `curriculum_{easy,medium,hard}.parquet` (in order) |
| | `dataset_shuffle` | **false** (curriculum ordering) |
| | `split_dataset_ratio` / `max_length` | 0.0 / 8192 |
| GSPO | `importance_sampling_level` | `sequence` |
| | `beta` | 0.0 (no KL) |
| | `epsilon` / `epsilon_high` | 3e-4 / 4e-4 |
| | `num_generations` | 16 |
| | `steps_per_generation` | 16 |
| | `num_iterations` | 1 (on-policy) |
| | `dynamic_sample` / `max_resample_times` | true / 3 (anti-collapse) |
| rollout | `use_vllm` / `vllm_mode` | true / `colocate` (in-process) |
| | `vllm_gpu_memory_utilization` | 0.5 |
| | `vllm_enforce_eager` | true |
| | `vllm_max_model_len` | 8192 |
| | `vllm_limit_mm_per_prompt` | `{"image": 1}` |
| | `sleep_level` | 1 (offload engine during optim step) |
| | `max_completion_length` | 2048 |
| | `temperature` / `top_p` / `top_k` | 1.0 / 0.9 / 20 |
| optim | `num_train_epochs` | 1 (one pass over the ramp) |
| | `per_device_train_batch_size` | 2 |
| | `gradient_accumulation_steps` | 8 |
| | `learning_rate` / `warmup_ratio` | 1e-6 / 0.05 |
| | `deepspeed` | zero2 (2B fits on 4×L40S; simpler/faster colocate sync — use zero3+offload if scaled up) |
| io | `save_steps` / `save_total_limit` / `logging_steps` | 200 / 3 / 5 |
| | `log_completions` / `report_to` | true / tensorboard |

**SLURM:** submit from `qure5` (`tsh ssh janhavi.prabhu@qure5.internal.qure.ai`); default 4×L40S,
24 h, `--exclude=qrnd12`; the sbatch activates `swift_infer` and sets the §1 env vars.

---

## 6. Smoke validation

A 4×L40S colocate smoke on the 4B CT checkpoint (`checkpoint-14000`) completed all 4 steps with
`EXIT CODE 0`:

```
vLLM visual_3d: installed 445 trained tensors ... built + loaded model.visual_3d on cuda:0
Train 1/4 ... 4/4  train_runtime 97.5s   memory ~53 GiB
```

This validated: from-source vLLM on the 535 driver, the `visual_3d` build + load, ZeRO-3 policy
attach, the `qwen3_5` template fix (images patchify, generation runs), the router + embedding reward
plumbing, and rewards logged. (Smoke `max_completion_length=512` caused all completions to clip →
`reward=0`; the real config uses 2048.)

---

## 7. Open items / future work

- **Safety modulation currently covers Report + ASP** (the fully-covered embedding tasks). KG (60%
  covered) and Comparison (pair-based, 0% covered) are not yet safety-modulated.
- **Comparison safety** would need pair-level tag completeness (both studies).
- **Safety→weight map is linear** (`w_emb *= safety`); could be made convex (`safety**2`) or given a
  floor for harsher down-weighting of sparse GT.
- **Sub-sample budgets** (800/tier/task) are deliberately small; raise for a larger run.
- **`qwen3_5_vl_ct` template** does not exist yet — needed only if CT *volumes* are fed to RL (2D
  X-ray RL uses `qwen3_5`).
- **Model wired:** `Qwen3.5-2B_stage6b_alltools/v0-20260906-173237/checkpoint-608` — a plain 2D
  Qwen3.5-2B (no `vision_3d_config`, patch 16, native `template=qwen3_5`), so the `visual_3d` path is
  a no-op here and `deepspeed: zero2` is used. The `visual_3d` machinery (§2.1) remains for future
  2D+3D checkpoints.
