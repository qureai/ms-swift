# Stage 7 — SEM latent-token embedding supervision

Stage 7 inserts K = 8 reserved tokens, `<|sem_0|>` … `<|sem_7|>`, immediately after every `<|im_start|>assistant\n` header. It trains their final hidden states so that a small pooled head, `sem_head.*`, predicts the frozen teacher embedding of that turn's ground-truth report. The teacher is `bb_23sep`, a fine-tuned Qwen3-Embedding-4B that produces 2560-d vectors.

Everything else matches stage 6c: the AR loss, the ASL loss on the ML-Decoder head, augmentation, liger, ZeRO-2 and flash-attention. The start point is the stage-6b checkpoint-608.

**Data (`MIX`).** The user's rule for stage 7 is *"use only mimic chexpert and rex gradient data"*, so the default is `MIX=mcr`: the stage-6c files filtered to rows whose images all come from MIMIC-CXR or CheXpert, plus ReXGradient-160K train/valid rows, all under `data_preparation/training_data/stage7_mcr/` ([§3.1](#31-data-mixmcr-default-and-mix6c)). `MIX=6c` selects the full stage-6c mixture and its own sidecar, unchanged.

```bash
# -1. data, once (CPU, ~4 min; already built on this host -- see stage7_mcr/README.md)
CUDA_VISIBLE_DEVICES="" python examples/train/liger/sem_latent/build_stage7_mcr_data.py
# 0. check everything, launch nothing
DRY_RUN=1 bash examples/train/liger/20_stage7_sem_latent.sh
# 1. teacher sidecar, once (8 GPUs, resumable; the launcher prints this exact command when it is missing)
mkdir -p /fastdata4/sahil_work/VLM_qure/data_preparation/training_data/sem_targets/stage7_mcr_rule_v1 && \
  GPUS="0 1 2 3 4 5 6 7" setsid nohup bash examples/train/liger/sem_latent/run_build_sem_targets.sh \
  > /fastdata4/sahil_work/VLM_qure/data_preparation/training_data/sem_targets/stage7_mcr_rule_v1/build.log 2>&1 < /dev/null & disown
# 2. train (prechecks in the foreground, then detached; log in $OUTPUT_DIR/train_<date>.log)
DETACH=1 bash examples/train/liger/20_stage7_sem_latent.sh
# 3. serve: swift infer from THIS repo's swift works as-is (pass --enable_thinking false); for native sglang/vLLM
#    make a -lm export. Any other swift build (upstream ms-swift, a site-packages copy, the Dockerfile.qwen3vl image)
#    serves a SEM checkpoint with ZERO SEM tokens and says nothing -- see §4.5 "Which swift".
python examples/deploy/export_lm_only.py <run>/checkpoint-N <run>/checkpoint-N-lm
```

**Status (2026-09-30).**
- **Sidecars.** The `MIX=6c` sidecar (`sem_targets/stage7_rule_v1`) was built on GPU: 8 shards, about 2 min wall. The default `MIX=mcr` sidecar (`sem_targets/stage7_mcr_rule_v1`) is **not built yet**. Only its CPU text collection exists, so the launcher refuses to start until step 1 has finished.
- **GPU validation so far.** That sidecar build, plus two 4-GPU × 30-step A/B smoke runs (SEM vs no SEM) that completed ([§7](#7-validation-so-far)). An 8-GPU smoke on the old 6c mixture hung.
- **Not done yet.** No full-length run, and no SEM checkpoint has been served.
- **Before a `MIX=mcr` run,** read the data-quality issues in [§8](#8-known-data-quality-issues-mixmcr), D1 in particular.

---

## 1. What and why

**The idea.** A report-generating VLM should carry a representation of *what the report will say* before it writes anything. Stage 7 makes that representation explicit and supervised:

- **SEM tokens before the caption.** A block of K prompt tokens sits right after every assistant header. Attention is causal, so their hidden states see the system prompt, the image, the user prompt and the history, but none of the answer. The last decoder layer's output at those K positions (post-norm, 2048-d each) is pooled by `SemanticHead`:
  - LayerNorm;
  - pooling over K: a learned attention query initialised to zero (so it starts as the mean), or plain `mean`, or `concat`;
  - Linear 2048 → 2048, SiLU, Linear → 2560;
  - L2 normalisation.

  The head has 9.45 M parameters. SEM labels are always -100: the tokens are never generated and never appear in decoded text, and the CE loss is untouched.
- **Frozen teacher.** The targets come from `bb_23sep`, and it never trains here.
  - Every target is embedded offline once, into a sidecar. Recipe: left padding, `<|endoftext|>` appended (mandatory; without it cos drops to 0.14–0.62), `.strip()`, last-token pooling, fp32 L2 norm, no instruction prompt, max_length 1024.
  - The teacher is fingerprint-pinned (`36741eaf96c44658…`) to the backbone the `lp23sep` tag probe was fitted on. That is why the probe diagnostic in [§4.4](#44-evaluate) is meaningful.
- **Which turns are supervised.** Only T1 report turns: `onto`, `fict`, `medrep`, and the template-G `sec_findings`, `sec_summary` and `sec_impression`. A turn qualifies only when the prompt prefix and the answer format agree.
  - The target text is one canonical string, `Findings: …\nSummary: …\nImpression: …`, built from the sections the turn itself contains. Complete Reasoning, Recommendations and Prominence_Score are dropped.
  - Findings and Impression are verbatim identical across families for the same film, so the per-film target is consistent.
  - Every other kind of turn gets a SEM block but no loss: cls, bbox, ASP, graph, sjson, T3 coordinate JSON, tool calls.
  - The rule lives in one pure module, `swift_plugins/sem_targets.py` (`RULE_VERSION = 'v1'`). The training plugin and the sidecar builder share it byte for byte, so a lookup can miss only if the data changed after the build.
- **Loss (fp32).** `z^` is the head output and `z*` the teacher target, both unit vectors in the uncentred teacher space. `μ` is the sidecar mean and `c(v) = normalize(v − μ)`.
  - `L_cos = mean(1 − ⟨c(z^), c(z*)⟩)` over the valid blocks.
  - `L_nce` is soft InfoNCE over the candidates `C` = this micro-batch's `c(z*)` plus a per-rank FIFO queue of 16,384 past `c(z*)`. The target distribution is `p = softmax(c(z*)·Cᵀ/τ_t)`, with τ = τ_t = 0.05. The optimised term is the **KL** `CE − H(p)`, where `CE = −Σ p·log_softmax(c(z^)·Cᵀ/τ)`. It has the same gradient as the CE, because `H(p)` has none, but it is 0 at the optimum. The CE carries `H(p)`, which grows with the queue fill (0.86 nats on average on real targets with a full queue), so it would drift `loss`, `eval_loss` and `sem_share` upward on its own. The CE is still logged, as `train/sem_nce_ce`.
  - Eval forwards score against their own micro-batch only, not the training queue, so `eval_loss` does not depend on how full the queue is. Retrieval over the whole eval set is `val_sem/*`.
  - `total = CE + ASL + 0.05·L_cos + 0.05·L_nce`.
  - A hard mode (`SEM_NCE_MODE=hard`) uses one-hot InfoNCE instead. It masks negatives that share the target key or the full image path, or whose teacher centred cos is above 0.9.
- **Why cos *and* InfoNCE.** Cosine alone is satisfied by a head that outputs the "average normal report" for everything: high cos, zero retrieval. InfoNCE forces the prediction to be closer to its own report than to others'.
- **Why soft targets.** Normal films collapse in teacher space: raw normal–normal cos is about 0.76–0.86 (map_data.md §4). One-hot InfoNCE would push near-identical normal reports apart as if they were negatives. Soft targets let the teacher's own similarity structure decide how much each candidate counts.
- **Why centring.** The teacher space is strongly anisotropic, with ‖μ‖ ≈ 0.70. A constant predictor `z^ = μ/‖μ‖` already scores raw cos 0.719, so a raw-cos loss has a trivial 72% solution. After centring with a fixed μ, that mean predictor scores centred loss ≈ 1, and mean off-diagonal centred cos is 0.03. `val_sem/cos_raw_meanpred` is logged as the baseline to beat.

**Why not read the latent at EOS / `<|im_end|>` of the report?**

1. **Timing.** At inference the report does not exist yet. The SEM block is computed in the prefill from image + prompt, so the latent is available before a single token is generated, for retrieval, consistency checks against the generated text, or gating. An EOS latent exists only after full generation.
2. **Train/serve mismatch and a shortcut.** Under teacher forcing, the EOS hidden state has read the ground-truth report. The loss could then be solved by embedding the text it just read (text-to-text distillation, no visual grounding). At inference it would read the model's own generation instead, which is a different input.
3. **Objective coupling.** `<|im_end|>` / EOS states are already CE-supervised (to end the turn and predict the next header). Tool-calling turns also have several end positions inside one merged turn. Fresh, dedicated rows avoid both problems.
4. **Free capacity.** The 2B checkpoints have 109 spare embedding rows (vocab_size 248320, tokenizer 248211). K ≤ 109 costs no resize, and a resize would re-tie `lm_head` ([§6](#6-backward-compatibility-and-escape-hatches)).

The teacher itself *does* pool at its own end-of-text token; that is its native recipe and is not the same question.

## 2. Exact placement and train/serve parity

```
<|im_start|>system\nYou are a helpful medical assistant.<|im_end|>\n
<|im_start|>user\n<vision>…</vision>Generate a detailed ontology report …<|im_end|>\n
<|im_start|>assistant\n<|sem_0|>…<|sem_7|><think>\n\n</think>\n\n{"Findings": …}<|im_end|>\n     <- training
<|im_start|>assistant\n<|sem_0|>…<|sem_7|><think>\n\n</think>\n\n                                <- inference prompt
```

- **Right after the header, before any `<think>`.** This is deliberate. The empty-think block appears in some places and not others, and not the same way in training and inference (a pre-existing quirk):
  - training adds it to every str-content assistant turn but never to merged tool-calling turns;
  - inference strips it from history and adds it only to the generation prompt.

  "header + SEM" is the only position that is identical in every mode. SEM ids are special tokens, so `header + SEM + <think>` tokenises to exactly header ids + SEM ids + think ids.
- **Every assistant header gets one block.** That includes history turns, tool-context headers, the generation prompt, and B3 tool rows. In B3 rows the block sits at the start of the merged turn, before the tool trace. The model must therefore predict the final report before it sees its own tool results, which is also exactly what inference can reproduce.
- **Core hook, not a plugin.** `swift/template/sem_latent.py` plus a 5-line hook (and 3 comment lines) at the top of `Template._encode_context_list` in `swift/template/base.py`. This is the one function every mode reaches: train, transformers/pt, sglang, vllm, lmdeploy. Three more lines in `Template.decode` strip SEM token strings from decoded responses. Everything is behind the same `if sem_str:` gate, so without `<|sem_0|>` nothing changes.
  - In sglang/vllm mode, `_encode_truncated` calls `Template._encode` directly, which bypasses any `Qwen3_5Template` patch.
  - Most eval harnesses import no plugin at all: in-process `SwiftInfer(... --load_args false)`, direct `TransformersEngine(ckpt)`, `-lm` exports without `args.json`.

  Activation is decided by the **tokenizer alone**: `<|sem_0|>` present means active. K is counted from the tokenizer (`<|sem_i|>` for i = 0, 1, … until one is missing) and is never read from an env var, because env vars are not restored at inference.
- **Native sglang / vLLM OpenAI servers** render the checkpoint's `chat_template.jinja` instead of swift's template.
  - The SEM plugin installs a SEM-aware jinja (`make_sem_chat_template`) on the processor and tokenizer at train start, so every saved checkpoint ships it.
  - `export_lm_only.py` re-applies it. It is idempotent: a leading SEM string in fed-back history is stripped before being re-added.
  - Token-id parity between HF `apply_chat_template` and swift's sglang-mode encode:
    - asserted in 32 of 32 cases in `tests/qure/test_sem_insertion.py`: 1-turn, multi-turn, history with think, history already containing SEM, a tool step, text and image, thinking on and off;
    - re-checked on 1-turn and 2-turn prompts while writing this README.
- **Weights at serve time.** SEM rows live in `embed_tokens` and survive every export. `sem_head.*` is dropped from `-lm` exports; sglang would skip it anyway. Its name contains none of the substrings sglang renames (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`, `in_proj_*`, `visual`, `language_model`, `embed_tokens`).

## 3. Files

| file | role |
|---|---|
| `examples/train/liger/20_stage7_sem_latent.sh` | Launcher: prechecks, `DRY_RUN`, `DETACH`, `RESUME`; the SEM plugin is listed last. |
| `examples/train/liger/token_sem.txt` | `<|sem_0|>` … `<|sem_7|>`, passed as the second `--new_special_tokens` file. `token.txt` is untouched. |
| `swift/template/sem_latent.py` + hook in `swift/template/base.py` | Insertion in every mode; helpers such as `get_sem_ids`, `count_sem_blocks`, `find_sem_blocks`, `sem_position_mask`, `strip_sem`; `make_sem_chat_template` / `install_sem_chat_template`. |
| `swift_plugins/sem_targets.py` | The target rule (T1 families, canonical text, `target_key` = the first 8 bytes of sha1), the leak guard, and `SidecarReader` (memory-mapped). Pure stdlib + numpy. |
| `examples/train/liger/sem_latent/build_sem_targets.py` | Offline teacher sidecar: collect, encode shards, merge. Resumable, fingerprint-checked. |
| `examples/train/liger/sem_latent/run_build_sem_targets.sh` | One detached shard per GPU, then merge. `MIX` picks the default `OUT` / `DATASETS` / `VAL_DATASETS`, as in the launcher. |
| `examples/train/liger/sem_latent/build_stage7_mcr_data.py` | Builds the `MIX=mcr` data (`stage7_mcr/`): MIMIC/CheXpert filter of the stage-6c files, ReXGradient rows, leakage assertions, `MANIFEST.json`, `README.md`. CPU only, deterministic. |
| `swift_plugins/sem_latent_plugin.py` | Training plugin: targets in encode, collator, `_post_encode` stash, `sem_head`, hooks, loss, optimizer group, SEM-row init, head restore, logging and eval. Installs nothing at inference. |
| `swift_plugins/qure_plugin_mode.py` | F1 `dedup_module` / `once`; F2 mode detection from `argv[0]`. |
| `swift_plugins/qure_untie_config_plugin.py` | F3, opt-in for any launcher. The SEM plugin applies the same function itself. |
| `examples/deploy/untie_lm_head.py` | F3 report tool for existing checkpoints (dry run by default). |
| `examples/deploy/export_lm_only.py` | F4 `-lm` serving export. |
| `tests/qure/test_*.py`, `tests/qure/conftest.py` | See [§10](#10-tests). `conftest.py` skips the directory on hosts without checkpoint-608 or the data. |
| `QURE_README.md` (repo root), `swift_plugins/README.md` | The fork-level overview (environment, F1–F4, tests, data governance) and the plugin table with load-order rules. |

### 3.1 Data: `MIX=mcr` (default) and `MIX=6c`

`MIX` sets the launcher's default `DATASETS`, `VAL_DATASETS` and `SEM_TARGETS_DIR`, and the sidecar builder's default `DATASETS`, `VAL_DATASETS` and `OUT`, so each MIX's default sidecar covers exactly its default files. An explicit `DATASETS`, `VAL_DATASETS` or `SEM_TARGETS_DIR` always wins.

| | `MIX=mcr` (default) | `MIX=6c` |
|---|---|---|
| train | `stage7_mcr/{A_TRAIN_120k,B3_TRAIN,15_stage5_headguided_TRAIN,17_refresher_multiturn_TRAIN,RET_diverse60k_TRAIN}.mcr.jsonl` + `stage7_mcr/REX_TRAIN.jsonl` | `stage6_toolcalling/A_TRAIN_120k`, `B3_TRAIN`, `15_stage5_headguided_TRAIN`, `17_refresher_multiturn_TRAIN`, `stage6_toolcalling/RET_diverse60k_TRAIN` |
| val | `stage7_mcr/A_VAL.mcr.jsonl`, `B3_VAL.mcr.jsonl`, `REX_VAL.jsonl` | `A_VAL`, `B3_VAL` |
| sidecar | `sem_targets/stage7_mcr_rule_v1` (not built yet) | `sem_targets/stage7_rule_v1` (built 2026-09-30) |
| `SKIP_BROKEN_IMAGES` | exported as `1` unless set | left as is |
| rows / epoch | 351,548 (229,476 MIMIC/CheXpert + 122,072 ReX), about 343 optimizer steps at batch 1024 ([§8](#8-known-data-quality-issues-mixmcr) D6) | 967,467, about 945 steps |

`stage7_mcr/` is written by `build_stage7_mcr_data.py`. Its `MANIFEST.json` holds every count, predicate, sha256 and the exact launcher strings; `README.md` there is the readable version. Nothing existing was modified.

- **MIMIC/CheXpert filter.** Each stage-6c file is copied line for line, in order, keeping only the rows where every image is under `/cxr_data/opensource_datasets/{mimic,chexpert}/`, after stripping the storage root and swift's `image_<W>x<H>_` prefix. Text-only rows are dropped; there are none.
  - Also dropped: rows that mix a MIMIC/CheXpert image with any other source, and internal re-exports of MIMIC/CheXpert films (`internal_data/training/images/images/{mimic.,mim.,chex.}*`). Those re-exports are not on disk here and carry no report turn.
  - Kept rows per file: A 30,428 / 120,202, B3 149,984 / 669,000, HG 16,177 / 71,074, RF 17,737 / 47,191, RET 15,150 / 60,000, A_VAL 454 / 1,900, B3_VAL 1,162 / 5,000. These match the scout census exactly.
  - All 121,238 distinct images exist under `/localstorage`, and a 256-image sample decodes (mode L).
  - The report families that survive are onto, fict, medrep and sec_*. **No row with a real box or crop survives**: MIMIC and CheXpert have no boxes, and every box row uses internal or segmed images. The B3 tool-calling rows survive (143,926).
  - **Among them are 5,018 `N7` rows** whose model-written `detect` result is always `{"boxes": []}`, so the bbox tool only ever sees empty answers ([§8](#8-known-data-quality-issues-mixmcr) D1).
  - About 75% of the rows in the unfiltered 6c mixture point at segmed or internal images that are not mounted on this host. swift silently replaces a missing image with 512×512 noise unless `SKIP_BROKEN_IMAGES=1`.
- **ReXGradient-160K.** `REX_TRAIN.jsonl` is built from the official TRAIN split only: 122,072 studies from 83,174 patients. `REX_VAL.jsonl` is 2,000 VALID studies, one per patient. The TEST split is never used.
  - **Image.** One per study: `png_hires/<key>.png`, the frontal pick upscaled to 8-bit with min side 1500 px. It is byte for byte the image and path the ckpt-608 public-test benchmark used. The raw 16-bit `deid_png` files would be saturated by swift's `.convert('RGB')`.
  - **Studies dropped:** lateral or oblique picks, non-chest studies (body part or description), de-identification tokens, and Findings that are junk or empty after cleaning.
  - **Findings duplicate cap.** Identical cleaned Findings are capped at 1,000 copies (`--rex_findings_cap`). Only the 12,953-copy normal macro exceeds it.
  - **Answers.** Every answer uses a stage-6c prompt string exactly (asserted against the filtered files) and a stage-6c answer format, all built from ONE per-study transform, so a study's canonical SEM target is identical in every family:
    - Findings become `* ` bullets, one per radiologist sentence, in the radiologist's order, with pure view/technique sentences removed.
    - The Impression stays prose. Recommendation sentences move to Recommendations; otherwise Recommendations is the fixed stage-6c sentence.
    - The cls turn is the Gemini tag extraction of the same report, run through the stage-6c `normalize_json_target`.
  - **Family mix**, one row per study (`--rex_mix`): fict 30%, cls→fict 20%, cls→two of {sec_findings, sec_impression, medrep} 20%, sec_findings 20%, medrep 10%. Studies with no usable Impression get sec_findings only.
  - **SEM targets per epoch.** 146,502 T1 turns, of which 140,321 are not leaked. The leaked ones are mostly medrep after sec_findings, which is masked on purpose and still CE-trained.
  - **ASL.** ReX images have no 132-tag labels in `cls_label_sidecar` (0 key collisions), so their ASL term is masked.
- **Leakage.** Every assertion ran on the rows in memory and again on the re-read files; all pass. Any violation aborts the build before anything is renamed into place.
  - Keys belong to their own split. Patient, accession and study UID are disjoint from the other two splits, and train patients are not in `audit_datasplit/rexgradient_clean_patients.txt`.
  - Nothing overlaps any scored test set: `rextest_gt.jsonl`, `rextest_keys_*.json`, `rex_test_clean_ids.json`, the ReXVQA test images or `rexgradient_eval/img8`.
  - No test image path appears, in any of its png_hires, png8, deid_png or img8 forms, and no test `GRDN*` id appears in any written line.
  - The MIMIC/CheXpert files contain 0 ReXGradient ids.
- **Overlap and contamination.**
  - A_VAL.mcr and B3_VAL.mcr share films with their train files, inherited from the 6c split: 262 of 504 and 988 of 1,189 images. `REX_VAL` is patient-disjoint from `REX_TRAIN`, so report `val_sem_unseen/*` and the REX_VAL numbers.
  - After training on REX_TRAIN, the ckpt-608 "all 160k" ReX generations (`reports_*`, `hr_*`, `r8_*`, `rex_chexbert.py` all_f1) are contaminated for the train split. Evaluate ReX on the public test split only.
  - The ReX style is taught under the fict, medrep, sec_* and cls prompts, not under the ontology prompt that the ckpt-608 benchmark used.
- **Licence.** ReXGradient-160K is licensed for non-commercial research only, with no redistribution. Keep `stage7_mcr/` internal and never push it to HF. A checkpoint trained on it is research-only.

## 4. Runbook

### 4.0 Dry run

```bash
cd /fastdata4/sahil_work/VLM_qure/vlm/swift
DRY_RUN=1 bash examples/train/liger/20_stage7_sem_latent.sh          # ~10-20 s; exit 0 = every precheck passed
```

The dry run prints the resolved env, then `[stage7] CMD: swift sft …`, a copy-pastable command, and launches nothing. A real launch runs the same prechecks:

- **Bash checks:**
  - The base checkpoint, every dataset file (a swift `#N` sampling suffix is allowed), both token files, every plugin file, and `cls_label_sidecar` exist.
  - `SEM_TARGETS_DIR/meta.json` exists. If not, the launcher prints the exact precompute command for *your* DATASETS / VAL_DATASETS.
  - The RESUME directory exists, if RESUME is set.
  - Root disk ≥ `ROOT_FREE_MIN_GB` and output disk ≥ `OUT_FREE_MIN_GB`.
  - The selected GPUs are idle. A real launch refuses GPUs using more than `GPU_BUSY_MIB`; the dry run only reports them. The script never kills anything.
  - `SWIFT_SEM_LATENT != 0`, and `QURE_PLUGIN_MODE` is unset or `train`.
- **Python checks** (CPU only, `CUDA_VISIBLE_DEVICES=""`, `python -P` so swift is imported exactly as the real launch imports it):
  - **Sidecar:** it opens with the plugin's own `SidecarReader`; `rule_version` matches; a teacher fingerprint is recorded.
  - **Coverage:** every train and val file was part of the build, with the same size and mtime, and the sidecar is not a `--limit` smoke build. `SEM_SIDECAR_CHECK=warn` downgrades these two checks to warnings.
  - **Knobs:** every `SEM_*` knob parses (the plugin's own `SemConfig`).
  - **Tokenizer:** the base tokenizer plus both token files go through swift's own `get_processor`. The check requires K contiguous SEM ids, K ≤ 109, and `len(tokenizer) ≤ config vocab_size`, so there is no resize.
  - **Core hook:** a real train-mode and sglang-mode encode through the working tree gives exactly one SEM block per assistant header, with labels -100. This check fails if the hook in `base.py` is missing; a negative control in the tests confirms it.
  - **Chat template:** it converts to the SEM-aware version.
  - **Checkpoint keys:** `ml_decoder_head.*` (23 in ckpt-608), `sem_head.*` (0, so this is the first SEM stage and the head starts fresh), `lm_head.weight` and the tie flags.

A complete sidecar that is wrong for your data (wrong rule_version, other files, a smoke build) is never overwritten by the builder. The precheck then prints a rebuild command into a new directory, `<dir>_rebuild_<date>`.

### 4.1 Precompute the teacher sidecar

```bash
# default OUT = $DATA_DIR/sem_targets/stage7_mcr_rule_v1 (MIX=mcr) | stage7_rule_v1 (MIX=6c) = the launcher's
# default SEM_TARGETS_DIR for the same MIX; pass MIX=6c to both scripts for the 6c mixture
mkdir -p <OUT> && GPUS="0 1 2 3 4 5 6 7" OUT=<OUT> setsid nohup bash \
  examples/train/liger/sem_latent/run_build_sem_targets.sh > <OUT>/build.log 2>&1 < /dev/null & disown
tail -f <OUT>/logs/shard_*.log                        # done when <OUT>/meta.json exists
```

The build has three stages:

1. **Collect**, once on CPU: unique canonical texts go to `<OUT>/collect/`.
2. **Encode**: one `setsid nohup` shard per GPU writes `parts/part_<i>.npz`, with chunk-level resume and OOM batch halving.
3. **Merge**: writes `keys.u64.npy`, `emb.f16.npy`, `mu.f32.npy`, `in_train.u8.npy`, `families.u8.npy`, `probe_logits.f16.npy`, `ntok.i32.npy`, `texts.jsonl`, and `meta.json` last. `meta.json` existing means the sidecar is complete.

Re-running is always safe. Finished shards and chunks are reused, and a running shard is waited on rather than duplicated. A chunk or part is reused only when its recorded provenance matches the current run: teacher fingerprint, dtype, max_length, probe head path and sha256, and RULE_VERSION. A re-run with another teacher, dtype or probe head re-encodes instead of mixing old rows into a sidecar whose meta records the new settings.

**Measured on CPU (collect only):**

| | `MIX=mcr` (default) | `MIX=6c` |
|---|---|---|
| rows scanned / rows with a target | 355,164 / 234,595 | 974,367 / 305,155 |
| unique targets | 208,788 | 218,505 |
| in train | 206,790 | 217,608 |
| in val | 2,693 (1,998 of them val-only) | 1,933 (897 of them val-only) |
| onto / fict / medrep | 66,326 / 57,901 / 25,578 | 174,883 / 14,127 / 7,959 |
| sections (findings / summary / impression) | 40,435 / 4,328 / 15,561 | 7,799 / 8,097 / 5,858 |
| turns onto / fict / medrep / findings / summary / impression | 93,481 / 67,581 / 29,470 / 48,344 / 4,726 / 23,361 | 269,337 / 14,135 / 8,904 / 9,161 / 8,960 / 9,013 |
| teacher tokens (EOT incl.) | 20.5 M, p50 89, p99 250, max 666 | 27.9 M, p50 123, max 490 |

- Nothing is truncated at 1024 in either mixture.
- The `MIX=mcr` collection is already cached in `sem_targets/stage7_mcr_rule_v1/collect/` (built on CPU 2026-09-30, 4 s). The GPU build reuses it while the data files keep their size and mtime.
- The sidecar is about 1.1 GB of fp16 for 6c, and about 1.07 GB for mcr.
- **Encode time.**
  - **Measured (6c, bf16, 27.9 M tokens):** each of the 8 shards took about 96 s (24–36 k tokens/s). From shard launch to the merged `meta.json` took about 2 min 12 s (`stage7_rule_v1/logs/`).
  - `encode_seconds: 767.9` in `meta.json` is the **sum** over the 8 shards, not wall time.
  - **Estimate (mcr, 20.5 M tokens):** about 1.5–2 min wall.
  - CPU fp32 runs at about 110–140 tokens/s, which would take about 45 h.

For a quick CPU smoke build (tiny, `--limit`), use `DEVICE=cpu LIMIT=24 OUT=<scratch>`. The launcher refuses such a sidecar unless `SEM_SIDECAR_CHECK=warn`.

A sidecar is tied to the dataset files: size and mtime are recorded and checked. If you change DATASETS, build a new sidecar. The launcher prints the command for your selection.

### 4.2 Train

```bash
DETACH=1 bash examples/train/liger/20_stage7_sem_latent.sh     # prechecks here, then setsid nohup … & disown
# or, foreground inside your own detach wrapper:
LOG=<run_dir>/train_$(date +%Y%m%d_%H%M%S).log
setsid nohup bash examples/train/liger/20_stage7_sem_latent.sh > $LOG 2>&1 < /dev/null & disown
```

- The launcher sets `oom_score_adj=1000` on itself, and torchrun, the ranks and the dataloader workers inherit it. The OOM killer therefore takes training before the box or the session.
- For the memory watchdog of the 6c launch, copy `data_preparation/training_data/stage6_toolcalling/launch_stage6c.sh` and change the script name.
- The default run (`MIX=mcr`) is 1 epoch over 351,548 rows at global batch 1024, about 343 optimizer steps (`MANIFEST.json` says 344 = ceil(rows / 1024); the HF trainer gives floor(5,493 / 16) = 343). It has 252,897 SEM-supervised (non-leaked) T1 turns, about 737 per step. `MIX=6c` is 967,467 rows, about 945 steps; 6c took about 31 h on the same data.
- `EVAL_STEPS=50` / `SAVE_STEPS=50` give about 7 evals and checkpoints per `MIX=mcr` epoch.
- Checkpoints are saved every 50 steps. Each is about 41 GB with optimizer state, and 3 are kept.
- The run directory is `$DATA_DIR/output/Qwen3.5-2B_stage7_sem_latent/v<N>-<date>/`.

**Resume:**

```bash
RESUME=<run_dir>/checkpoint-N IGNORE_DATA_SKIP=1 DATA_SEED=43 DETACH=1 bash examples/train/liger/20_stage7_sem_latent.sh
```

- lazy_tokenize resumes replay the skipped rows through the loader, which is slow. Prefer `IGNORE_DATA_SKIP=1` with a fresh `DATA_SEED`, as for 6b.
- `sem_head` and the SEM rows should come back with the rest of the module state: the head is attached at `prepare_model`, before the optimizer, so the saved and live module trees match. This is not verified on GPU.
- The InfoNCE queue is not checkpointed, so it starts empty again.

**To continue from 6c instead of 6b:** `BASE_MODEL=<6c ckpt-945>`. Its config, tokenizer and jinja are byte-identical to 6b's, so every check holds.

### 4.3 Monitor

The run logs to TensorBoard / wandb (`<run_dir>/runs`) and to `logging.jsonl`. mldecoder's `ar` / `asl` entries are unchanged: its hook runs before the SEM hook, so `ar` excludes the SEM term. `loss` is the total.

| key | meaning | what to expect |
|---|---|---|
| `train/sem_cos_loss` | L_cos = 1 − centred cos | Starts ≈ 1.0 (random head). Must fall. |
| `train/sem_cos_centred`, `train/sem_cos_raw` | Mean centred / raw cos of z^ vs z* | Raw cos is only meaningful above ≈ 0.72, the mean-predictor baseline. Centred cos starts near 0. |
| `train/sem_nce_loss` (= `train/sem_kl` in soft mode), `train/sem_nce_ce` | The optimised soft-InfoNCE KL (CE − H(p)), and the CE for reference | The KL is 0 for a perfect student whatever the queue size. The CE grows with the candidate count until the per-rank queue is full; do not compare early CE values with late ones. Estimated fill time from the census: `MIX=mcr` about 737 / 8 = 92 valid targets per rank per step, so about 180 steps to fill 16,384; `MIX=6c` about 41 per rank per step, so about 400 steps. While the head is still at chance, the KL sits near ln(number of candidates): measured 3–5 with ≤ 55 queued entries in the [§7](#7-validation-so-far) smoke. Hard mode: `sem_nce_loss` is the one-hot CE. |
| `train/sem_share` | SEM term as % of the total loss | **Watch this.** Measured in the 30-step smoke: mean 47%, 26–73% after step 10, with the queue holding ≤ 55 entries. Estimate for a full queue with a head that has not learnt: 0.05·(ln 16k − H(p)) ≈ 0.44, about as large as CE+ASL or larger. See the loss-balance guidance below this table. |
| `train/sem_valid_frac` | Supervised SEM blocks / all SEM blocks | Should be stable. Most blocks belong to non-report turns and are unsupervised by design. |
| `train/sem_miss_rate` | Sidecar misses / (valid + misses) | ≈ 0 by construction: the builder collects a superset of what the plugin sees (872/872 hits on 2,100 real rows). Above about 0.1% means the data or the rule changed; rebuild. |
| `train/sem_leak_frac` | Blocks masked because the target was already verbatim in earlier context | About 2.9% of supervised turns (template-G later sections, some head-prior rows). |
| `train/sem_mismatch_frac` | Rows whose SEM block count ≠ assistant turns, or whose target errored | ≈ 0. Non-zero means a literal `<|im_start|>assistant\n` inside user/tool text, or a truncated row. |
| `train/sem_w_cos`, `train/sem_w_nce`, `train/sem_queue_fill`, `train/sem_detached` | Weights, queue size, and head-only warm-up flag | `sem_detached` = 1 for the first `SEM_DETACH_STEPS` (20) steps. The head trains on detached hidden states, so the LM is not dragged by a random head. |

The plugin also logs `[sem-latent] step=… valid a/b blocks … cos_c=… raw=… | sem term … of total …` every `SEM_LOG_EVERY` training micro-batches.

**Loss balance (guidance from the smoke runs).**
- In the 30-step A/B, the SEM term reached about 0.2–0.3 once the queue had entries (0.05·L_cos ≈ 0.05 plus 0.05·KL with KL 3–5). That is about half of the logged total loss (`train/sem_share` mean 47%, 26–73% after step 10), while `train/ar_loss` (CE) was unchanged at 0.1700 vs 0.1701 without SEM.
- Thirty steps are far too few to show whether a dominant SEM term eventually pulls the LM.
- **During the run,** after the `SEM_DETACH_STEPS` warm-up, watch `train/sem_share` together with `train/ar_loss` (and `eval_token_acc`).
- **If `train/ar_loss` rises after the warm-up** (compared with a no-SEM reference, or with its own level at the end of the warm-up), lower `SEM_W_NCE` to about 0.02, or schedule it down (`SEM_W_NCE_START`, `SEM_W_NCE_END`, `SEM_W_NCE_DECAY_STEPS`). `SEM_W_COS` contributes only about 0.05 at chance level.

### 4.4 Evaluate

**During training (every `EVAL_STEPS`, gathered across ranks):**

- `val_sem/*` covers every eval target:
  - `cos_centred` and `cos_raw`;
  - `cos_raw_meanpred`, the constant-μ baseline;
  - `r1`, `r5`, `r10`, which are duplicate-aware (a hit is the same key or teacher centred cos > 0.95);
  - `median_rank`, `margin` (cos to own target − mean cos to the others), `margin_centred`, `n` and `n_gallery`.
- `val_sem_unseen/*` repeats the same metrics on targets whose text never occurs in a training file (sidecar `in_train == 0`). **Report these.** The MIMIC/CheXpert val files overlap training, as A_VAL and B3_VAL always did:
  - `MIX=mcr`: 695 of the 2,693 val texts are also train texts. A_VAL.mcr shares 262 of 504 images with train, B3_VAL.mcr 988 of 1,189. `REX_VAL` is patient-disjoint from `REX_TRAIN`.
  - `MIX=6c`: 1,036 of the 1,933 val texts are also train texts, and map_data.md measured 57% of B3_VAL T1 texts in train.
- `val_sem_probe/macro_auc` and `val_sem_probe/n_tags_scored`:
  - The `lp23sep/head_xray_bce.pt` tag probe is applied to the uncentred z^.
  - Labels are `probe(z*) ≥ threshold`.
  - The macro AUC is taken over tags with at least `SEM_PROBE_MIN_POS` (50) positives.
  - This asks whether the SEM prediction carries the findings the teacher sees in the real report.
  - When no tag reaches `SEM_PROBE_MIN_POS` positives, only `n_tags_scored = 0` is logged and there is no `macro_auc`. That was the case in the 30-step smoke, with 60 eval targets.
- `eval_loss` now includes the SEM terms, so it is **not comparable** with the eval_loss of 6b/6c. Within one run it is comparable across evals: the eval SEM term uses the eval micro-batch as its only InfoNCE candidates, never the training queue. swift picks the "best" checkpoint by `loss`, which is `eval_loss` here.

**Offline analysis.** Rank 0 dumps `<run_dir>/eval_predictions/sem_step{N:06d}.npz` with `z_hat` (f16), `keys`, `img_keys`, `in_train`, `rank`, `cos_centred` and `cos_raw`. Join the dump to the targets like this:

```python
import numpy as np, swift_plugins.sem_targets as st
d = np.load('<run_dir>/eval_predictions/sem_step000900.npz'); r = st.SidecarReader('<SEM_TARGETS_DIR>')
z_star = r.rows(r.lookup(d['keys'])).astype(np.float32)        # teacher targets, same order as d['z_hat']
```

**Generation quality.** Existing harnesses work unchanged on a stage-7 checkpoint **as long as they import this repo's swift**, because SEM insertion is tokenizer-gated in this fork's swift core. That covers `swift infer` subprocess runs, the in-process `SwiftInfer` workers under `output/`, `TransformersEngine` scripts, and the teacher-forced `compose_eval/eval_family_loss.py`, all run from the `ms-swift-may-s4` env or with `PYTHONPATH=<repo>`. Those harnesses live outside this repo, or in `examples/deploy/`, which is not in this branch. Any other swift build inserts nothing and warns about nothing: see "Which swift" in [§4.5](#45-serve--export), and add the canary there to harnesses. Always pass `--enable_thinking false`; with thinking on, the core logs a one-time warning. Compare against 6b ckpt-608 served the same way, and keep F3 in mind ([§6](#6-backward-compatibility-and-escape-hatches)).

**Backward-compatibility regression.** Run the tests in [§10](#10-tests).

### 4.5 Serve / export

**Which swift.** SEM insertion is a feature of **this fork** (`swift/template/sem_latent.py` plus the hook in `base.py`). The checkpoint carries no insertion logic for swift's own template backend, so the swift build that serves it decides:

| serving path | SEM block in the prompt? |
|---|---|
| this repo's swift: the `ms-swift-may-s4` env (editable install), or any env with `PYTHONPATH=<repo>` | yes, every mode |
| upstream `ms-swift`, the `ms-swift-may` env (it imports its site-packages swift), the `paper_vlm/serving/docker/Dockerfile.qwen3vl` image (upstream `ms-swift[all]`; its `COPY swift/` override is commented out) | **no, and silently.** Only a restored `sem_latent_plugin` fails loudly there (ModuleNotFoundError); in-process workers, `swift deploy` on a `-lm` export and `TransformersEngine` scripts do not. |
| any swift with `--template_backend jinja` | yes: the checkpoint ships the SEM-aware `chat_template.jinja` (verified: 8 SEM ids) |
| native sglang / vLLM OpenAI server on a `-lm` export | yes (SEM-aware jinja) |

**Canary.** Put this in every harness that serves a stage-7 checkpoint. It needs nothing from this fork, so it also catches a wrong swift build:

```python
sem0 = tokenizer.convert_tokens_to_ids('<|sem_0|>')
if isinstance(sem0, int) and sem0 >= 0 and sem0 != tokenizer.unk_token_id:   # a SEM checkpoint
    assert sem0 in prompt_input_ids, 'this swift build does not insert SEM tokens (see sem_latent/README.md §4.5)'
```

`prompt_input_ids` is the first request's encoded prompt: `template.encode(...)['input_ids']`, or `apply_chat_template(...)` for a native server.

**swift (pt or sglang backend).** A stage-7 checkpoint's tokenizer already contains the SEM tokens, and `args.json` restores `new_special_tokens` and the plugin list. Every restored plugin detects `swift infer` and installs nothing (F2). For example:

```bash
swift infer --model <run_dir>/checkpoint-N --val_dataset <in.jsonl> --result_path <out.jsonl> \
    --enable_thinking false --system 'You are a helpful medical assistant.' --max_length 28000 \
    --temperature 0 --infer_backend sglang --sglang_dp_size 8 --use_hf true
```

**Native sglang / vLLM OpenAI server.** Make a `-lm` export first:

```bash
python examples/deploy/export_lm_only.py <ckpt> <ckpt>-lm --dry-run     # plan: dropped keys, files
python examples/deploy/export_lm_only.py <ckpt> <ckpt>-lm               # writes <ckpt>-lm/export_manifest.json
```

The export:

- drops `ml_decoder_head.*`, `asl_loss.*` and `sem_head.*`, plus optimizer, rng and trainer state and `args.json`;
- sets `tie_word_embeddings: false` at both levels;
- rewrites the chat template to the SEM-aware jinja;
- writes a manifest.

Then run `python -m sglang.launch_server --model-path <ckpt>-lm …` (env `sg-lang`) and send `chat_template_kwargs: {"enable_thinking": false}`.

**Check that a checkpoint inserts SEM** (verified with an in-memory SEM tokenizer; parity was True on 1-turn and 2-turn prompts):

```python
from swift.model.register import get_processor
from swift.template import get_template
from swift.template.sem_latent import get_sem_ids, count_sem_blocks
proc = get_processor('<ckpt>'); sem = get_sem_ids(proc)                  # len(sem) == 8
t = get_template(proc, template_type='qwen3_5', enable_thinking=False, default_system='You are a helpful medical assistant.')
t.set_mode('sglang')
msgs = [{'role': 'system', 'content': 'You are a helpful medical assistant.'},
        {'role': 'user', 'content': 'Generate a medical report of the given chest X-ray.'}]
ids = t.encode({'messages': msgs})['input_ids']
j = proc.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=True)
j = j['input_ids'] if hasattr(j, 'keys') else j                         # BatchEncoding in transformers 5
print(count_sem_blocks(ids, sem), list(j) == list(ids))                   # 1 True  (native-server jinja == swift)
```

## 5. Knobs

### 5.1 Launcher (`20_stage7_sem_latent.sh`; every value is `${VAR:-default}`)

| knob | default | notes |
|---|---|---|
| `BASE_MODEL` | `…/Qwen3.5-2B_stage6b_alltools/v0-20260906-173237/checkpoint-608` | Any storage root is accepted ([§11](#11-paths-on-this-host)). |
| `RUN_NAME` / `OUTPUT_DIR` | `Qwen3.5-2B_stage7_sem_latent` / `$DATA_DIR/output/$RUN_NAME` | |
| `MIX` | `mcr` | `mcr` = MIMIC + CheXpert + ReXGradient only; `6c` = the stage-6c mixture. It sets the defaults of the next three rows ([§3.1](#31-data-mixmcr-default-and-mix6c)). `mcr|6c` only; anything else is refused. |
| `DATASETS` | `MIX=mcr`: `stage7_mcr/{A_TRAIN_120k,B3_TRAIN,15_stage5_headguided_TRAIN,17_refresher_multiturn_TRAIN,RET_diverse60k_TRAIN}.mcr.jsonl` and `stage7_mcr/REX_TRAIN.jsonl`. `MIX=6c`: A_TRAIN_120k, B3_TRAIN, 15_stage5_headguided_TRAIN, 17_refresher_multiturn_TRAIN, RET_diverse60k_TRAIN | B3 = B2 tool positives + N negatives, so the 6b tool-calling data is retained, filtered to MIMIC/CheXpert under `mcr`. Space-separated; a `#N` suffix is allowed. Setting it overrides `MIX`'s files. |
| `VAL_DATASETS` | `MIX=mcr`: `stage7_mcr/A_VAL.mcr.jsonl`, `B3_VAL.mcr.jsonl`, `REX_VAL.jsonl`. `MIX=6c`: A_VAL, B3_VAL | |
| `SEM_TARGETS_DIR` | `MIX=mcr`: `$DATA_DIR/sem_targets/stage7_mcr_rule_v1`. `MIX=6c`: `$DATA_DIR/sem_targets/stage7_rule_v1` | Same as `run_build_sem_targets.sh`'s default `OUT` for the same `MIX`. |
| `SKIP_BROKEN_IMAGES` | `MIX=mcr`: exported as `1`. `MIX=6c`: left as is | `1`: an image that fails to load raises, and swift's `LazyLLMDataset` drops the row and fetches another (after 10 failures in a row the run stops). Unset: `load_image` returns 512×512 random noise and the row trains on it. Every `stage7_mcr` image was checked to exist at build time. An explicit value always wins (e.g. `SKIP_BROKEN_IMAGES=0`). |
| `SEM_SIDECAR_CHECK` | `strict` | `warn` downgrades the coverage and `--limit` checks to warnings. Rule version and fingerprint problems always refuse. |
| `RESUME`, `IGNORE_DATA_SKIP` | unset, 0 | `--resume_from_checkpoint`, `--ignore_data_skip true`. |
| `DATA_SEED`, `BS`, `GAS`, `EPOCHS`, `NUM_WORKERS` | 42, 8, 16, 1, 12 | Global batch = NPROC × BS × GAS = 1024. |
| `LR`, `ALIGNER_LR`, `VIT_LR` | 4e-6, 5e-5, 2e-6 | As 6c. The SEM embedding rows train at `LR`. |
| `MAX_LENGTH`, `EVAL_STEPS`, `SAVE_STEPS`, `SAVE_TOTAL_LIMIT`, `LOGGING_STEPS` | 28000, 50, 50, 3, 5 | |
| `REPORT_TO` | `tensorboard wandb` | |
| `GPUS`, `NPROC_PER_NODE` | `0,…,7`, count of `GPUS` | |
| `EXTRA_PLUGINS` | — | Inserted before `sem_latent_plugin.py`, which always stays last. |
| `EXTRA_ARGS` | — | Appended to `swift sft`, e.g. `--max_steps 20`. It comes last, so it overrides the defaults. The Python precheck parses the **final** argv with swift's own parser and runs the SEM plugin's refusals on it, so `--loss_scale all` or `--template_backend jinja` fails in the precheck, not after the model load. |
| `DDP_TIMEOUT` | 10800 | `--ddp_timeout` in seconds: the NCCL watchdog timeout that turns a rank desync into an abort. swift's default of 18,000,000 s (about 208 days) would hang instead. It must exceed the longest legitimate single wait: a rank-0 checkpoint save (about 41 GB with optimizer), a full eval pass, or a first-time (cold Arrow cache) dataset build behind swift's rank barrier. Raise it for new DATASETS. |
| (auto) `--remove_unused_columns false` | added when the sidecar holds override texts | `sem_targets` / `sem_target` override columns reach encode only with this flag, because swift's loader drops non-standard columns. The launcher adds it when `meta.json` `n_texts_per_family.override > 0`, unless EXTRA_ARGS sets the flag. The default mixture has no override columns, so the default command is unchanged. |
| `DRY_RUN`, `DETACH`, `PRECHECK` | 0, 0, 1 | See [§4](#4-runbook). `PRECHECK=0` skips only the Python checks. |
| `ROOT_FREE_MIN_GB`, `OUT_FREE_MIN_GB` | 3, 150 | Every cache and TMPDIR are redirected off root. |
| `GPU_BUSY_MIB`, `ALLOW_BUSY_GPUS` | 2048, 0 | |
| `OOM_SCORE_ADJ` | 1000 | `''` leaves it alone. |
| `CACHE_ROOT`, `CACHE_TMPDIR`, `CONDA_ENV` | `…/cache_stage6`, `/dev/shm/cache_stage6_tmp`, `ms-swift-may-s4` | The 6c cache is shared, so its Arrow cache of these files is reused. |
| `AR_WEIGHT`, `ASL_WEIGHT` | 1.0, **0.05 constant** | When `ASL_WEIGHT` is set, mldecoder bypasses the START/END schedule. |
| `MLD_RESTORE_HEAD`, `MLD_FREEZE_HEAD` | 1, 0 | The ML-Decoder head is restored and keeps training. |
| `QURE_*` | as 6c | `QURE_AUG_SKIP_BBOX=1`, `QURE_AUG_BBOX_MODE=photometric`, `QURE_W_TRUE=6.0`, `QURE_W_COORD=2.0`, … |

### 5.2 SEM plugin (`swift_plugins/sem_latent_plugin.py`; the header has the full semantics)

| knob | default | meaning |
|---|---|---|
| `SEM_TARGETS_DIR` | required | Sidecar directory. |
| `SEM_W_COS`, `SEM_W_NCE` | 0.05, 0.05 | Constant weights. |
| `SEM_W_{COS,NCE}_{START,END,DECAY_STEPS,DECAY_SHAPE}` | unset, const, 1000, cosine | Optional schedule on the optimizer step (cosine, linear or exp). |
| `SEM_TAU`, `SEM_TAU_T` | 0.05, = `SEM_TAU` | Student and teacher temperatures. |
| `SEM_NCE_MODE` | `soft` | `hard`: one-hot InfoNCE with false-negative masks. |
| `SEM_FN_THRESH` | 0.9 | Hard mode: mask negatives with teacher centred cos above this. |
| `SEM_QUEUE` | 16384 | Per-rank FIFO of past targets; 0 means in-batch only. |
| `SEM_NCE_GATHER` | 0 | 1 adds other ranks' targets as candidates (an `all_gather` every micro-step). |
| `SEM_POOL`, `SEM_HEAD_HIDDEN`, `SEM_HEAD_DTYPE` | attn, 2048, bfloat16 | Head shape. Changing it breaks a later `SEM_RESTORE_HEAD`. |
| `SEM_HEAD_LR` | 2e-4 | Own optimizer group, weight decay 0. With HF's default optimizer (no `--vit_lr` / `--aligner_lr`), which puts every trainable param into its decay / no-decay groups, the head params are moved into this group with a warning. |
| `SEM_DETACH_STEPS` | 20 | Head-only warm-up on detached hidden states. |
| `SEM_RESTORE_HEAD`, `SEM_FREEZE_HEAD` | 1, 0 | Restore `sem_head.*` from `BASE_MODEL` (none in 6b, so it starts fresh); freeze = no optimizer group. |
| `SEM_INIT_EMBED`, `SEM_INIT_NOISE`, `SEM_INIT_TOL`, `SEM_SEED` | 1, 0.5, 0.05, 1234 | Re-initialise the SEM rows of `embed_tokens` only if they are still the spare vector (tolerance-based: the ckpt-608 spare rows are equal only to bf16 rounding). New value = mean(trained rows) + N(0,1)·std·0.5, seeded, identical on every rank. `lm_head` rows are unchanged. |
| `SEM_ONLINE_TEACHER`, `SEM_ONLINE_CACHE`, `SEM_TEACHER_PATH` | 0, 50000, meta `teacher_path` | 1 encodes cache misses with the teacher on the rank's GPU, kept off the module tree and fingerprint-checked. |
| `SEM_SKIP_LEAKED`, `SEM_SKIP_LOSS_FALSE` | 1, 1 | Leak guard; turns whose final answer is `loss:false`. |
| `SEM_PROBE_HEAD`, `SEM_PROBE_MIN_POS`, `SEM_DUP_THRESH` | `lp23sep/head_xray_bce.pt`, 50, 0.95 | Eval diagnostics. |
| `SEM_LOG_EVERY`, `SEM_EVAL_PRED_DIR` | 20, `<output_dir>/eval_predictions` | |
| `SEM_UNTIE_CONFIG`, `SEM_CHAT_TEMPLATE` | 1, 1 | F3 untie on the in-memory config; install the SEM-aware jinja so checkpoints ship it. |

### 5.3 Sidecar builder (`run_build_sem_targets.sh`)

| knob | default | notes |
|---|---|---|
| `MIX` | `mcr` | Picks the defaults of `OUT`, `DATASETS` and `VAL_DATASETS`, exactly as the launcher does. |
| `OUT` | `MIX=mcr`: `$DATA_DIR/sem_targets/stage7_mcr_rule_v1`. `MIX=6c`: `$DATA_DIR/sem_targets/stage7_rule_v1` (built; the builder never overwrites a complete sidecar) | |
| `DATASETS`, `VAL_DATASETS` | Same defaults as the launcher, for the same `MIX` | Keys seen in `DATASETS` get `in_train=1`. |
| `DEVICE`, `GPUS`, `DTYPE` | cuda, `0 … 7`, bf16 (fp32 on CPU) | One shard per GPU. |
| `MAX_LENGTH`, `BATCH_TOKENS`, `MAX_BATCH`, `CHUNK_TEXTS` | 1024, 32768 (4096 on CPU), 256, 4096 | |
| `LIMIT`, `WRITE_TEXTS`, `WORKERS`, `THREADS`, `POLL`, `LOG_DIR` | —, 1, 96, 64, 30, `$OUT/logs` | |
| `TEACHER`, `PROBE_HEAD`, `HEADS_JSON` | `bb_23sep`, `lp23sep/head_xray_bce.pt`, `lp23sep/heads.json` | The build refuses on a fingerprint mismatch unless `--allow_fingerprint_mismatch`. |

### 5.4 Core

| knob | default | meaning |
|---|---|---|
| `SWIFT_SEM_LATENT` | enabled iff `<|sem_0|>` is in the vocabulary | `0` disables insertion even with SEM tokens. Use it only for A/B tests: a SEM-trained checkpoint expects the block. |

## 6. Backward compatibility and escape hatches

- **Core insertion is a strict no-op without `<|sem_0|>`.** `get_sem_str` returns `''` and the same list objects flow on.
  - Every checkpoint before stage 7 encodes byte-identically: 72 of 72 records identical to the pre-change `base.py` on every key, across train, pt, sglang and vllm modes and 9 real rows (tool, `loss:false`, role `tool`, bbox, cls, multi-turn).
  - An independent re-check at final verification gave 420 of 420 identical encodes: 30 random real rows from 9 files, loaded through swift's `load_dataset`, × 14 mode variants including lmdeploy, `Template.decode` included.
  - With plugins, `QURE_PLUGIN_DEDUP=0` reproduces the pre-change plugins on 30 of 30 seeded train encodes, pixel values included.
  - The cost is 1.9 µs per call.
  - No existing launcher, checkpoint, dataset or `token.txt` was modified. The SEM tokens come from a separate `token_sem.txt`.
- **The four bug fixes are on by default**, each with a switch back to the legacy behaviour:

| fix | what changed | default effect | legacy switch |
|---|---|---|---|
| **F1** plugin double import | Every plugin calls `dedup_module(__name__, '<name>')` first, so it is one module object under `X` and `swift_plugins.X`; patches are `once()`-guarded; `qure_stage3` guards the augmentation module that is actually installed. A plugin that raises during import (the load-time albumentations check) leaves no half-initialised alias: `drop_failed_import` pops it before a retry, so the load still fails as before F1. | Non-bbox images are augmented **once** and bbox rows get **photometric-only** aug. (In 6b/6c, non-bbox images were augmented twice and bbox rows still got geometric aug.) | `QURE_PLUGIN_DEDUP=0` reproduces 6c exactly. |
| **F2** inference never detected under the `swift` CLI | `qure_plugin_mode` reads `argv[0]` (`swift/cli/<verb>.py`). `export --to_cached_dataset` stays train so cls labels are still baked into the cache. The flag is matched the way swift's parser matches it: `--to-cached-dataset`, unambiguous abbreviations such as `--to_cached`, `=value`, last one wins, and `RAY_SWIFT_ARGS`. | `swift infer/deploy/eval/export` install no train-only patch. Encodes were verified identical (input_ids / labels / pixel hashes, pt and sglang). | `QURE_PLUGIN_MODE=train` (`=infer` forces inference). The launcher refuses `infer`. |
| **F3** train/serve lm_head mismatch | Checkpoints store a separate, trained `lm_head.weight` but say `tie_word_embeddings: true`. HF trains untied, while sglang 0.5.10 and vLLM 0.20.2 tie and serve lm_head = embed_tokens. The SEM plugin (and the opt-in `qure_untie_config_plugin.py`) set the flag false on the in-memory config at both levels, so stage-7 checkpoints are saved untied and a later resize cannot re-tie. Training is bit-identical. | Serving a stage-7 checkpoint with sglang/vLLM uses the **trained** lm_head, whereas 6b/6c checkpoints served embed_tokens. The difference is mean \|Δ\| ≈ 0.3% of \|lm_head\|, max 0.0038. Part of any served stage-7 vs 6c difference is this fix, not SEM. | `SEM_UNTIE_CONFIG=0`, `QURE_UNTIE_CONFIG=0`. List `qure_untie_config_plugin.py` last; every `swift_plugins/` wrapper now tolerates it anywhere, but a third-party plugin reading `TunerMixin.prepare_model.__func__` must come before it. The SEM plugin finds `untie_tied_config` even without the repo root on `sys.path`, and raises if it cannot. For old checkpoints, `untie_lm_head.py <ckpt>` is a dry run by default; `--apply` writes `config.json` and keeps `config.json.tied_bak` (restore by copying it back). **It has not been applied to any existing checkpoint; that is your call.** |
| **F4** `-lm` export script missing | `export_lm_only.py` reproduces `checkpoint-608-lm` (643 → 618 keys) and adds untie + SEM jinja + a manifest. `--overwrite` first removes what an earlier export left and this one does not write, such as an `args.json` from a `--keep-args-json` run, which swift would restore training args from, or stale shards. Unknown files stay, with a warning. | — | `--no-untie`, `--no-sem-template`, `--keep-args-json`, `--drop-prefix`. |

The copies under `paper_vlm/plugins/` were not changed. They pick up F2 through `swift_plugins.qure_plugin_mode`, but they have no dedup.

## 7. Validation so far

Everything below was measured on 2026-09-30 on `qrnd3` (8 GPUs), env `ms-swift-may-s4`.

**Runs:**

| run | setup | outcome |
|---|---|---|
| Teacher sidecar, `MIX=6c` | `run_build_sem_targets.sh`, 8 shards (one per GPU), bf16, max_length 1024 | 218,505 targets, 27.9 M tokens, 0 truncated, ‖μ‖ 0.709. About 96 s per shard; about 2 min 12 s wall from shard launch to the merged `meta.json`. |
| Teacher precision | bf16 vs fp32 teacher embeddings | cos ≥ 0.9998; centred off-diagonal cos 0.002 |
| 8-GPU smoke, **old 6c mixture** | Launcher with `DATASETS` = the 5 stage-6c files sampled with `#N` (1,100 rows); BS 2, GAS 2, `--max_steps 20`, `SEM_DETACH_STEPS=5`, default `DDP_TIMEOUT` (10800 s) | **Hung at step 3** (the last logged step is 2/20), with all ranks spinning in NCCL. That mixture had 178 unreadable segmed/internal images, replaced by random noise (`SKIP_BROKEN_IMAGES` was unset). py-spy stack dumps were refused (no permission). **Cause not proven.** The run was stopped by hand. |
| A/B, 4 GPUs × 30 steps | MIMIC/CheXpert-only rows sampled from the 6c files (1,160 train / 128 val); BS 2, GAS 2 (global batch 16); ZeRO-2 + Liger + flash-attn; eval + save at steps 15 and 30; `DDP_TIMEOUT=600`; `SEM_DETACH_STEPS=5`; the 6c sidecar with `SEM_SIDECAR_CHECK=warn`. **A** = this launcher. **B** = the same `swift sft` command without `token_sem.txt` and without `sem_latent_plugin.py`. | **Both completed, no hang** |

**A/B results:**

| metric | A: SEM | B: no SEM |
|---|---|---|
| CE (`train/ar_loss`), mean over 30 steps | 0.1700 | 0.1701 |
| `eval_token_acc`, step 15 / 30 | 0.9526 / 0.9529 | 0.9527 / 0.9528 |
| `eval_loss`, step 15 / 30 | 0.253 / 0.247 (**includes the SEM terms**) | 0.210 / 0.206 |
| logged `loss`, mean over 30 steps | 0.411 | 0.265 |
| `train/sem_share`: mean / range after step 10 | 47% / 26–73% | – |
| `train/sem_kl` (soft InfoNCE), steps 10–30 | 2.1–5.2 (3–5 typical); queue 20 → 55 entries | – |
| `train/sem_cos_loss` | 0.90–1.07 (chance ≈ 1) | – |
| `train/sem_valid_frac` (supervised / all SEM blocks), mean | 0.24 | – |
| `train/sem_miss_rate`, `train/sem_mismatch_frac` | 0 at every step | – |
| `val_sem/cos_centred`, step 15 / 30 | 0.008 / 0.011 (chance) | – |
| `val_sem/r10`, step 30 | 0.23 (60 targets, 58-entry gallery; random ≈ 10/58 = 0.17) | – |
| `val_sem_probe/n_tags_scored` | 0 of 131 (no tag reached 50 positives) | – |
| `train_runtime` (30 steps) / peak `memory(GiB)` | 281.9 s / 63.2 | 283.3 s / 60.9 |

**Checkpoints of run A** (`checkpoint-15`, `checkpoint-30`):
- `config.json` has `tie_word_embeddings: false` at the top level and in `text_config` (F3).
- `model.safetensors.index.json` lists 7 `sem_head.*` and 23 `ml_decoder_head.*` tensors.
- `chat_template.jinja` is the SEM-aware one.
- The tokenizer has the SEM tokens at ids 248211..248218.
- `args.json` lists the SEM plugin. At `swift infer` it is restored and installs nothing (F2).
- The startup line read: `[sem-latent] K=8 SEM tokens, ids 248211..248218 (contiguous=True) | embedding rows 248320 (ckpt vocab 248320; no resize) | lm_head tied to embed_tokens: False | sidecar …stage7_rule_v1: n=218,505`.

**What this shows:**
- The SEM hooks, the loss, the optimizer group and the eval `all_gather_object` work under DeepSpeed ZeRO-2 + Liger + flash-attention on 4 ranks.
- Checkpoints come out complete.
- CE and token accuracy did not move within 30 steps.

**What it does not show:**
- The head stayed at chance. The smoke supervised only about 4 blocks per optimizer step (global batch 16), and 5 of the 30 steps were the detached warm-up.
- 30 steps at lr 4e-6 cannot show whether SEM helps or hurts the LM.
- The 8-GPU hang is unexplained. The hung run differs from the A/B in rank count (8 vs 4), data (6c rows including 178 noise images vs MIMIC/CheXpert rows with none) and `DDP_TIMEOUT` (10800 vs 600).

**Still untested on GPU:**
- an 8-GPU run that completes;
- the `MIX=mcr` data and sidecar;
- a full-length run;
- `SEM_NCE_GATHER=1` over NCCL;
- the online teacher on GPU;
- resume with `sem_head`;
- serving a SEM checkpoint (swift sglang/pt, native sglang/vLLM on a `-lm` export);
- the F3 serving effect.

**Before the first full run:**
1. Build the `MIX=mcr` sidecar ([§4.1](#41-precompute-the-teacher-sidecar)) and pass `DRY_RUN=1`.
2. Repeat the smoke on 8 GPUs with the `mcr` files, with a short NCCL timeout and the debug env the A/B used, so a hang aborts within minutes and leaves a dump:
   ```bash
   M=/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/stage7_mcr
   PYTHONFAULTHANDLER=1 TORCH_DISTRIBUTED_DEBUG=DETAIL TORCH_NCCL_DESYNC_DEBUG=1 TORCH_NCCL_TRACE_BUFFER_SIZE=2000 \
   TORCH_NCCL_DUMP_ON_TIMEOUT=1 DDP_TIMEOUT=600 BS=2 GAS=2 NUM_WORKERS=4 LOGGING_STEPS=1 SEM_LOG_EVERY=1 \
   SEM_DETACH_STEPS=5 EVAL_STEPS=15 SAVE_STEPS=15 SAVE_TOTAL_LIMIT=1 REPORT_TO=tensorboard \
   RUN_NAME=Qwen3.5-2B_stage7_smoke_mcr OUTPUT_DIR=<scratch run dir> \
   DATASETS="$M/A_TRAIN_120k.mcr.jsonl#300 $M/B3_TRAIN.mcr.jsonl#500 $M/REX_TRAIN.jsonl#300" \
   VAL_DATASETS="$M/A_VAL.mcr.jsonl#48 $M/REX_VAL.jsonl#48" \
   EXTRA_ARGS="--max_steps 30 --save_only_model true" DETACH=1 bash examples/train/liger/20_stage7_sem_latent.sh
   ```
   A smoke checkpoint is about 5 GB with `--save_only_model true`; delete `<scratch run dir>` afterwards.
3. In the full run, apply the loss-balance guidance in [§4.3](#43-monitor).

## 8. Known data-quality issues (`MIX=mcr`)

These come from an independent verification of `stage7_mcr/`. All seven checks pass: no ReXGradient test-split leakage; every image is from MIMIC, CheXpert or ReXGradient; the filtered files are byte-exact subsets. Nothing below has been fixed; the data files, `MANIFEST.json` and the builder are unchanged.

| # | severity | issue | numbers | options |
|---|---|---|---|---|
| **D1** | medium | **The bbox tool only ever learns to return empty boxes.** B3 `N7` rows keep the model-written `detect` result, `<tool_response>{"boxes": []}</tool_response>`, as an assistant message with no `loss` flag, so it is trained on. Every row with a real box or crop was dropped, because those films are internal images, not MIMIC/CheXpert. Stage 7 therefore shows `detect` only empty answers and can erode the bbox skills learned in 6b. | 5,018 `N7` rows in `B3_TRAIN.mcr.jsonl`, 96 in `B3_VAL.mcr.jsonl`; 5,114 `boxes` occurrences, all `[]`; 20,200 dropped crop references | **(a)** Drop the `N7` rows, or only their `detect` sub-turns, from the MCR copies. The rows carry `"family": "N7"`, a column swift's loader drops. There is no builder flag for this today, so it needs a builder change or a filtered copy. Then rebuild the sidecar, because the sidecar pins each file's size and mtime. **(b)** Accept it, and re-run the bbox evaluation after training against ckpt-608. |
| D2 | low | Some ReX "chest" studies carry a report about another exam (foot, tibia, wrist, venous Doppler, head ultrasound, abdomen). Source noise, not leakage; the test split has the same rate. | 247 `REX_TRAIN` rows (0.20%), 3 `REX_VAL`; the test split has 38 per 10k | A text filter on the report (no chest vocabulary → drop) would remove them. |
| D3 | low | Leftover text in ReX answers: a de-identification run of underscores, explicit mm/dd/yyyy dates (the stage-6c MIMIC answers never contain dates, so the model will learn to write them), and "recommend…" sentences kept inside the Impression | 1 underscore row; 1,789 train + 15 val rows with dates; 42 fict rows | Strip dates and underscore runs in the ReX transform, then rebuild the data and the sidecar. |
| D4 | info | MIMIC/CheXpert rows dropped because they point at internal re-exports of those films (`internal_data/.../images/{mimic.,chex.}*`), which are not mounted on this host. Correct and conservative under the MIMIC/CheXpert/ReX-only rule. | 4,964 train rows: A 1,538, B3 333, 15_stage5 2,355, RET 738 | Recoverable by remapping the paths: 200/200 sampled `mimic.<id>` names exist as MIMIC jpgs. |
| D5 | info | The report-writing prompts will now mostly produce ReXGradient-style reports. | ReX outnumbers MIMIC/CheXpert about 10:1 on fict (60,857 vs 5,747) and Findings (42,925 vs about 4.7k) turns. `onto` stays MIMIC/CheXpert only (92,891 turns). ReX uses fict as the first image turn 36,513 times; 6c never did. 35% of training rows (ReX) have no ASL labels. | Evaluate MIMIC/CheXpert and ReX separately. `onto` is the one report prompt still trained only on MIMIC/CheXpert answers. |
| D6 | cosmetic | Two counts differ only by definition. | `optimizer_steps_at_1024` = 344 is ceil(rows / 1024); the HF trainer gives floor(5,493 / 16) = 343. The collect manifest's `n_texts_per_family_in_train` (fict 57,194, medrep 25,277) counts a text as in-train if it appears there under any family; a per-family recount gives 57,187 and 25,266. | None needed. |

## 9. Known limitations

- **Only partly validated on GPU** ([§7](#7-validation-so-far)).
  - **Measured:** the 6c sidecar build, and a 4-GPU × 30-step A/B under ZeRO-2 + Liger + flash-attention, with eval gather and checkpoint save.
  - **An 8-GPU smoke on the old 6c mixture hung at step 3,** cause not proven.
  - **Untested on GPU:**
    - an 8-GPU run that completes;
    - the `MIX=mcr` sidecar and data;
    - a full-length run;
    - `SEM_NCE_GATHER=1` over NCCL;
    - the online teacher (only tested fp32 on CPU);
    - resume with `sem_head`;
    - every serving path for a SEM checkpoint.

  **What was verified on CPU:**
  - ckpt-608 real weights: forward + backward, loss equal to w·L, gradients reaching the head and the SEM rows;
  - the toy end-to-end composition with the stage-6c plugin order;
  - every encode path;
  - this launcher, through a fake `swift`.

  **Do a short smoke run first:** see the recipe at the end of [§7](#7-validation-so-far), and check the `[sem-latent]` startup lines.
- **The bbox tool sees only empty boxes under `MIX=mcr`** ([§8](#8-known-data-quality-issues-mixmcr) D1). Decide on D1 before a long run.
- **History-turn think-block mismatch (pre-existing).** Training renders history as `assistant\n<think>\n\n</think>\n\nA1`, while inference renders `assistant\nA1`. B3 tool rows train with no empty think at all, while inference always adds one. SEM does not fix this; it only sits before it, which is identical everywhere.
- **SEM rows train at the LLM learning rate (4e-6).** They are initialised at the trained-row mean plus 0.5·std noise, and at that lr they will barely move. The latent is carried mostly by the transformer and the head (2e-4). The `lm_head` SEM rows are left as the spare vector; they are never label targets. There is no logit bias against generating SEM ids. The measured max P(any SEM) per generation position on ckpt-608 weights is about 1e-6 (untied) to 9e-6 (tied). `Template.decode` strips SEM token strings from swift's decoded responses, so an emitted SEM id never ends up in fed-back history as a stray block. Native sglang/vLLM servers skip special tokens by default.
- **Eval/val overlap.** For MIMIC/CheXpert, only `val_sem_unseen/*` measures generalisation: a held-out-by-film val split does not exist yet. `REX_VAL` is held out by patient, but its SEM metrics are pooled with the MIMIC/CheXpert val rows in `val_sem/*`.
- **ReXGradient rows (`MIX=mcr`).**
  - They carry no 132-tag labels, so the ASL term is masked for them. That is about 35% of training rows.
  - The frontal pick includes 21,838 train studies whose view is `UNKNOWN`. Some of those may be laterals; the public test split has the same mix.
  - The ReX style is learned under the fict, medrep, sec_* and cls prompts. The ckpt-608 benchmark used the ontology prompt, so evaluate ReX with fict (or cls → fict), on the public test split only; the "all 160k" numbers are contaminated.
  - The Findings duplicate cap (1,000) lowers the share of the most common normal macro. Normal prevalence in REX_TRAIN is therefore a few points below the test split.
  - The licence is non-commercial research only.
- **`eval_loss` includes the SEM terms** and is not comparable with 6b/6c. It is comparable across the evals of one run, because the eval term never uses the training queue.
- **B3 tool rows.** The block is at the start of the merged turn, so SEM predicts the final report from image + prompt + prior only, before the tool trace. That is harder than predicting it from inside the turn, but it is the only placement inference can reproduce with a static prefix.
- **Leak guard.** It masks turns whose Findings are already verbatim in earlier context (2.9%), and only exact matches after whitespace and JSON-escape normalisation.
- **Queue.** It is per rank, not checkpointed, and needs about 400 steps to fill, so L_nce and KL are not stationary early.
- **Refused configurations.** The plugin refuses these at `prepare_model`, and the launcher precheck refuses them on the final argv:
  - packing, padding_free and sequence parallel;
  - `--template_backend jinja`, which would label the SEM tokens or insert none;
  - a `--loss_scale` whose base strategy is `all` (`all`, `all+X`). Training would insert no SEM block, because the headers carry loss, while inference inserts one after every header. The core also warns once about such a training encode on a SEM tokenizer, even without the plugin.
  - a non-binary `--loss_scale` (`qure_graded`, `hermes`, …), `--loss_type`, `--label_smoothing_factor > 0`, `--enable_dft_loss` and `--enable_channel_loss`. With any of them swift computes the loss outside the model, so `out.loss` is None and the SEM term would silently train nothing. As a backstop, the loss hook raises if a training micro-batch with valid SEM blocks ever returns `loss=None`.

  Not refused, but note:
  - `--cached_dataset` must be re-exported with the plugin loaded;
  - `truncation_strategy left/right` can cut a block (the row is then unsupervised); the default `delete` is safe.
- **One image identity per row.** The first image is used (all T1 turns are single-image). Image keys are set only through the Qwen3.5 template's encode wrapper.
- **Override columns** `sem_targets` / `sem_target` reach `encode()` only with `--remove_unused_columns false`. The launcher adds the flag when the sidecar holds override texts, and the plugin warns when they would be dropped.
- **Stray headers.** A literal `<|im_start|>assistant\n` inside user or tool text also gets a block, and the row is then unsupervised (it shows up in `sem_mismatch_frac`).
- **K is fixed by the tokenizer.** Changing K needs a new `token_sem.txt`, a fresh head (`SEM_RESTORE_HEAD=0`) and a check that K ≤ 109.
- **No inference-time SEM head.** `sem_head.*` is dropped from `-lm` exports and no plugin reads it at serve time. Getting z^ at inference would need a pt-engine hook like `mldecoder_infer_plugin`; sglang's `return_hidden_states` is not wired through swift.
- **Teacher prompt budget.** The canonical target is at most 490 teacher tokens in the current data, so the 1024 cap never bites. Raw sjson or repcmp text would, but those families are excluded.
- **Thinking mode is warned about, not forced off.** Pass `--enable_thinking false` everywhere; every `examples/deploy` script already does.

## 10. Tests

All tests run on CPU:

```bash
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=/fastdata4/sahil_work/VLM_qure/vlm/swift
PY=/home/ubuntu/miniconda3/envs/ms-swift-may-s4/bin/python
```

| command | covers | result (final verification, 2026-09-30) | wall / peak RSS |
|---|---|---|---|
| `$PY tests/qure/test_sem_insertion.py` (~100 s) | Byte-identical encodes without SEM (72/72 vs pre-change code); one block per header with SEM, labels -100, mrope; jinja parity 32/32; `SWIFT_SEM_LATENT=0`; thread safety; `Template.decode` strips SEM only when SEM is active; the loss-scale-all warning; silent lookup on a Tekken (MistralCommonBackend) tokenizer | 53 checks passed (pytest: 7) | ~104 s / 5.9 GB |
| `$PY tests/qure/test_sem_targets.py` (`SEM_TEST_E2E=1` for the teacher end-to-end) | T1 rule vs the full census (precision 1.0, recall 0.99982); sidecar round-trip; builder resume, including provenance-checked chunk / part reuse | 20 passed | ~34 s / 24.2 GB with `SEM_TEST_E2E=1` |
| `$PY tests/qure/test_plugin_fixes.py --slow` (~6 min) | F1 probes in both import orders, and failed imports that stay loud; F2 on the real CLI and every `--to_cached_dataset` spelling pinned against swift's parser; F3 CPU load of ckpt-608 untied, the untie plugin listed first, the SEM untie without the repo root on `sys.path`; F4 synthetic export and `--overwrite`; test-module env hygiene | 25 passed | ~6.7 min / 9.1 GB |
| `SEM_TEST_ONLINE=1 $PY tests/qure/test_sem_plugin.py` (~3 min) | Loss math vs a float64 reference (soft L_nce = KL); head, queue, detach; eval without the training queue; the ZeRO-2 fallback's grad-ready order; the optimizer group move; training-arg refusals; loss=None raising; real-template targets; toy end-to-end and plugin composition | 157 checks / 30 tests passed | ~168 s / **122.5 GB** (the two toy end-to-end scenarios; the rest < 25 GB) |
| `$PY tests/qure/test_sem_integration_cpu.py` | ckpt-608 real weights forward/backward with SEM | 11 checks passed (pytest: 1) | ~53 s / 23.9 GB |
| `$PY tests/qure/test_stage7_launcher.py` (~3.5 min) | This launcher: `bash -n`; missing, stale, wrong-rule, smoke and uncovered sidecars; dry-run command contents; the real path runs `swift sft` with the dry-run argv (fake swift); `DETACH`; guards; overrides; `--ddp_timeout`; final-argv refusals; the automatic `--remove_unused_columns false`; negative control for the core-hook check; `MIX` defaults (mcr files + `stage7_mcr_rule_v1` + `SKIP_BROKEN_IMAGES=1`, `MIX=6c` = the 6c files + `stage7_rule_v1`, overrides win, bad `MIX` refused) | 14 passed | ~214 s / 1.3 GB |

`$PY -m pytest -q -p no:cacheprovider tests/qure` collects 97 tests (96 passed at final verification; `test_mix_defaults` was added after it and passes). Under pytest, use `QURE_SLOW=1` instead of `--slow`.

- **Output directories.** Set `SEM_TEST_OUT` / `QURE_TEST_OUT` / `STAGE7_TEST_OUT` per file. The defaults point at the build session's scratch directory.
- **Pre-change backup.** `test_sem_insertion.py` (a) needs the pre-change `base.py` (`SEM_TEST_BACKUP`); rebuild it from git as shown in [`QURE_README.md` §7](../../../../QURE_README.md#7-tests).
- **RAM.** `test_sem_plugin.py` needs a host with about 125 GB free RAM.

`tests/qure/conftest.py` leaves the directory out of collection when this host's checkpoint or data are missing (`QURE_TESTS_FORCE=1` collects anyway). Each module applies its process env (`CUDA_VISIBLE_DEVICES=''`, `IMAGE_MAX_TOKEN_NUM`, `QURE_PLUGIN_MODE`, …) only while its own tests run, and restores it afterwards, never at import.

## 11. Paths on this host

- **Storage roots.** The user wrote `/localstorage/sahil_work/tag_extractor/lp23sep`, but on this host the bundle is `/fastdata4/sahil_work/tag_extractor/lp23sep`. `/localstorage` exists but has no `sahil_work`, and `/fastdata3` is absent. The launcher, `sem_targets.resolve_storage_path` and the builder all try a path as given, then with its root swapped among `/localstorage`, `/fastdata3` and `/fastdata4`, as `swift/template/vision_utils.py` does for images. Either spelling works.
- **Teacher:** `/fastdata4/sahil_work/tag_extractor/bb_23sep`, the 23_sep checkpoint-900, fingerprint `36741eaf96c44658a2dc62b309dbb3aa3eccadcf62301163a729f198364ce728` = `lp23sep/heads.json` `backbone_fingerprint`.
- **Probe:** `/fastdata4/sahil_work/tag_extractor/lp23sep/head_xray_bce.pt`, 131 tags, standardised linear head fitted at 512 tokens. Texts shorter than 512 tokens embed identically at 512 and 1024.
- **Base checkpoint:** `/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/output/Qwen3.5-2B_stage6b_alltools/v0-20260906-173237/checkpoint-608`. It holds 643 keys, including 23 `ml_decoder_head.*`; `len(tokenizer)` is 248211 and the embedding has 248320 rows. With `token_sem.txt` the SEM ids are 248211..248218 and there is no resize.
- **Data (`MIX=mcr`):** `/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/stage7_mcr/` (see `MANIFEST.json` there). The ReXGradient images are `/localstorage/sahil_work/ReXGradient-160K/png_hires/<key>.png`, which is the form the ckpt-608 benchmark used. On this host they resolve to `/fastdata4/sahil_work/...`, an NFS mount of host .14's `/localstorage`, through swift's storage-root fallback. That is about 85 GB read over NFS per epoch.
- **Sidecar defaults:** `MIX=mcr`: `/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/sem_targets/stage7_mcr_rule_v1`, collection cached, not encoded yet. `MIX=6c`: `.../sem_targets/stage7_rule_v1`, built 2026-09-30. The run directory is `/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/output/Qwen3.5-2B_stage7_sem_latent/`.
- **Disk:** `/fastdata4` was at 99%, with 294 GB free, on 2026-09-30 after the smoke runs. Three kept checkpoints need about 125 GB, and the launcher refuses to start with less than `OUT_FREE_MIN_GB` (150) free.
