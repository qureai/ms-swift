# Qure fork of ms-swift: what is added on top of upstream

This repository is ModelScope's [ms-swift](https://github.com/modelscope/ms-swift) with Qure's chest X-ray VLM work on top. The Qure changes are:
- a set of training and inference plugins (`swift_plugins/`);
- a few small, gated patches to swift core;
- the stage-7 "SEM latent token" training recipe;
- tests that pin the whole thing down on CPU.

The models are Qwen3.5-2B checkpoints (stages 2 to 6c, then stage 7). They are trained with `swift sft`, served with `swift infer` / sglang, and they report on chest X-rays.

This file is the entry point for a teammate who has never seen the fork. Depth lives in:
- [`examples/train/liger/sem_latent/README.md`](examples/train/liger/sem_latent/README.md): stage 7 (method, runbook, every knob, validation, limitations).
- [`swift_plugins/README.md`](swift_plugins/README.md): the plugin table, load order, and the dedup and mode conventions.
- The long header docstring of each file. Every plugin and script documents its own knobs; this file only summarises them.

Contents:
1. [What this branch is](#1-what-this-branch-is)
2. [Directory map (what is in this branch)](#2-directory-map-what-is-in-this-branch)
3. [What is NOT in this branch](#3-what-is-not-in-this-branch)
4. [Environment](#4-environment)
5. [Bug fixes F1-F4](#5-bug-fixes-f1-f4)
6. [Backward-compatibility guarantees](#6-backward-compatibility-guarantees)
7. [Tests](#7-tests)
8. [Stage-7 quickstart](#8-stage-7-quickstart)
9. [Data governance](#9-data-governance)

---

## 1. What this branch is

```
upstream ms-swift  1654fa9ae  "[bugfix] fix peft>=0.19 target_parameters (#9483)"
      |
      +- 75a2f8114  (Qure, on main)  EVAL_README.md, medical_eval/run_api.sh
      +- 081654857  (Qure, on main)  storage-root image resolution in base.py / vision_utils.py, ASP tokens in
      |                              token.txt, qure_full_4b.sh, 11_June_internvl3_5_2b_full_data.sh, a notebook
      |
      +- commit 1   (this branch)    earlier, uncommitted Qure work: sglang logprob / grammar serving fixes,
      |                              image-path + broken-image handling, transforms_qure.py, the 132 tag tokens,
      |                              swift_plugins/ as they were BEFORE the F1-F4 fixes
      +- commit 2   (this branch)    SEM latent tokens (core hook + plugin + launcher + sidecar builder),
                                     bug fixes F1-F4 applied to every plugin, export/untie tools, tests/qure/,
                                     these READMEs
```

- **Branch:** `stage7-sem-latent`, in the private repo `qureai/ms-swift-qure`.
- **Why two commits:** commit 1 records the plugins exactly as the stage-6 runs used them. `git diff <commit 1> <commit 2> -- swift_plugins/` is therefore this branch's whole plugin change set, in the existing plugins plus the new files. In the existing plugins it consists of:
  - the F1 dedup blocks and `once()` guards;
  - the F2 mode detection;
  - `qure_stage3`'s augment-module resolution;
  - the `getattr(TunerMixin.prepare_model, '__func__', …)` change;
  - header notes.
- **Where it runs:** host `qrnd3` in the `ms-swift-may-s4` conda env ([§4](#4-environment)). Almost every default path points at `/fastdata4/sahil_work/...`, where the checkpoints and data live. On another host the tests skip themselves ([§7](#7-tests)).

## 2. Directory map (what is in this branch)

| path | what it is | notes |
|---|---|---|
| `swift_plugins/` | Qure training and inference plugins, loaded with `--external_plugins` | One line per plugin below; full table in [`swift_plugins/README.md`](swift_plugins/README.md). No `__init__.py` (namespace package): the repo root must be on `PYTHONPATH`. |
| `swift/template/sem_latent.py` + hook in `swift/template/base.py` | SEM latent-token insertion, in swift core | A strict no-op unless the tokenizer contains `<|sem_0|>`. See the note after this table. |
| `swift/template/transforms_qure.py` | Port of the foundation-model RandAugment pipeline (albumentations geometric + torchvision intensity) | Imported only by `qure_augment_plugin.py` / `qure_augment_internvl_plugin.py`. It imports `data.utils.transform_utils` from `FM_ACCELERATE_REPO`. Needs albumentations 1.x (1.3.1 in the env). |
| `swift/template/vision_utils.py` + `Template._load_image` in `base.py` | Image-path resolution and broken-image handling | See the note after this table. |
| `swift/infer_engine/sglang_engine.py`, `swift/infer_engine/protocol.py`, `swift/arguments/base_args/generation_args.py` | sglang serving fixes | See the note after this table. |
| `swift/rollout/multi_turn.py` | `GRPOVllmEngine` import made optional | vLLM is not installed in `ms-swift-may-s4`; GRPO still runs with `--use_vllm false`. |
| `examples/train/liger/token.txt` | `|<ASP>|`, `|</ASP>|` and the 132 finding tags (`<PNEUMONITIS>`, ...) | The Qure launchers, the stage-7 one included, pass it as `--new_special_tokens`. Unchanged by this branch's SEM work. |
| `examples/train/liger/token_sem.txt` | `<|sem_0|>` .. `<|sem_7|>` | Stage 7 passes it as a second `--new_special_tokens` file. |
| `examples/train/liger/20_stage7_sem_latent.sh` | Stage-7 launcher | Prechecks, `DRY_RUN`, `DETACH`, `RESUME`, `MIX=mcr` (default) or `6c`. Built on the structure of the stage-6c launcher, which is not in this branch. |
| `examples/train/liger/sem_latent/` | Stage-7 support: `README.md`, `build_sem_targets.py` + `run_build_sem_targets.sh` (offline teacher sidecar), `build_stage7_mcr_data.py` (the `MIX=mcr` data) | Start at its [README](examples/train/liger/sem_latent/README.md). |
| `examples/deploy/untie_lm_head.py` | F3 report / fix tool for existing checkpoints | Dry run by default. `--apply` rewrites `config.json` and keeps a `config.json.tied_bak` backup. |
| `examples/deploy/export_lm_only.py` | F4 `-lm` serving export | Drops the head tensors and training state; untie + SEM jinja; writes a manifest. |
| `tests/qure/` | 6 test files + `conftest.py`, CPU only | [§7](#7-tests). |
| `.gitignore` | Ignores local inference and analysis output dirs | `vlm_inference_analysis/`, `report_multiturn_140k/`, `report_compare_20k/`, `report_compare_nodule20k/`, `eval140k_internvl/`, `paper_vlm_release/`, `data_qure/`, `.gradio/`, `examples/train/liger/_smoke_subset.jsonl` |
| `EVAL_README.md`, `medical_eval/`, `qure_full_4b.sh`, `11_June_...sh`, the notebook | From main's earlier Qure commits | Not touched by this branch. The working tree had a further local edit to `qure_full_4b.sh`, which is not included. |

**Notes on the core changes:**
- **SEM hook (`sem_latent.py` + `base.py`).**
  - `base.py` gets 5 code lines (+3 comment lines) at the top of `Template._encode_context_list`, the one function every encode mode reaches. `Template.decode` gets 3 lines that strip SEM strings from responses. Plus one import.
  - `sem_latent.py` also carries the chat-template helpers (`make_sem_chat_template`, `install_sem_chat_template`) that give native sglang/vLLM servers a SEM-aware `chat_template.jinja`.
- **Image-path resolution (`vision_utils.py` + `Template._load_image`).**
  - A path under `/localstorage`, `/fastdata3` or `/fastdata4` resolves to whichever root has the file.
  - A file named `image_<W>x<H>_<name>` loads `<name>` and resizes it to W×H.
  - An image that fails to load becomes 512×512 random RGB noise, with a `[WARN] Failed to load image` print. With `SKIP_BROKEN_IMAGES=1` it re-raises instead, and swift's `LazyLLMDataset` drops the row. The stage-7 launcher sets this under `MIX=mcr`.
  - Upstream `load_image` has none of this and simply raises.
- **sglang serving fixes (`sglang_engine.py`, `protocol.py`, `generation_args.py`).**
  - `--token_ids_logprob` / `RequestConfig.token_ids_logprob`: per-position logprobs of fixed token ids. It is sglang only; use it, for example, to score ` true`/` false`.
  - sglang logprobs converted to swift's format, in full and in streaming responses; the old code returned `logprobs=None`.
  - `json_schema` / `regex` / `ebnf` and `structured_outputs_regex` are passed to sglang as grammar constraints.
  - `images` → `image_data` remap in the non-stream path.
  - `SGLANG_EXTRA_SERVER_ARGS='{"k": v}'` passes extra `ServerArgs` fields through, including `skip_tokenizer_init: false`, which grammar backends need.

**The plugins, in one line each** (the knobs listed are the main ones; the header of each file has all of them):

| plugin | purpose | loaded when | key env |
|---|---|---|---|
| `mldecoder_joint_plugin.py` | ML-Decoder head (132 tags) on the Qwen3.5 vision tower; ASL folded into the model loss next to the AR CE | 6b/6c/7 training, listed first | `AR_WEIGHT`, `ASL_WEIGHT` (constant) or `ASL_WEIGHT_START/_END/_DECAY_STEPS/_DECAY_SHAPE`, `MLD_HEAD_LR`, `CLS_LABEL_SIDECAR`, `FM_ACCELERATE_REPO` |
| `mldecoder_joint_internvl_plugin.py` | The same head for InternVL3.5 (taps `mlp1`) | InternVL training | as above, plus `MLD_HEAD_MAX_TOKENS` |
| `mldecoder_infer_plugin.py` | Re-attaches the trained head at inference and dumps per-image logits | `swift infer --infer_backend pt` | `MLD_CLS_OUT`, `MLD_TOP_K` |
| `qure_augment_plugin.py` | RandAugment (`transforms_qure`) on Qwen3.5 training images | training | `QURE_AUG_PROB`, `QURE_AUG_CROP_MIN/MAX`, `QURE_AUG_IM_SIZE`, `QURE_AUG_PHOTO_PROB` |
| `qure_augment_internvl_plugin.py` | The same for InternVL | InternVL training | `QURE_AUG_*` |
| `qure_ce_weight_plugin.py` | Token-id-weighted CE through Liger's fused `ce_weight` (true ×6, JSON structure down-weighted) | training with `--use_liger_kernel true` | `QURE_W_TRUE/_FALSE/_FORMAT/_DEFAULT/_COORD` |
| `qure_graded_loss_scale.py` | Registers the non-binary loss_scale `qure_graded` | when `--loss_scale qure_graded` is wanted | `QURE_W_*` (incompatible with stage 7: the SEM plugin refuses non-binary loss scales) |
| `qure_stage3_plugin.py` | Wrapper add-on: ML-Decoder head restore / freeze, bbox-safe augmentation guard | 6b/6c/7 training, after mldecoder_joint and qure_augment | `MLD_RESTORE_HEAD`, `MLD_FREEZE_HEAD`, `QURE_AUG_SKIP_BBOX`, `QURE_AUG_BBOX_MODE` |
| `force_presence_plugin.py` | Forces `"presence": true/false` by a per-finding threshold on P(true) during generation | inference, pt backend, batch 1 | `MLD_FORCE_TAU`, `MLD_FORCE_TAU_JSON`, `MLD_FORCE_OUT` |
| `record_presence_plugin.py` | Records P(true) at every `"presence":` slot without changing logits | inference, pt backend | `MLD_REC_OUT` |
| `qure_plugin_mode.py` | Not a plugin: train/infer mode detection (F2) and `dedup_module` / `once` (F1), imported by all of the above | always, indirectly | `QURE_PLUGIN_MODE`, `QURE_PLUGIN_DEDUP` |
| `qure_untie_config_plugin.py` | F3: saved configs say `tie_word_embeddings: false` | opt-in, any launcher (stage 7 gets the same thing from the SEM plugin) | `QURE_UNTIE_CONFIG` |
| `sem_latent_plugin.py` | Stage-7 SEM supervision: targets, `sem_head`, loss, optimizer group, logging and eval | stage-7 training, listed LAST | `SEM_TARGETS_DIR` (required), `SEM_W_COS`, `SEM_W_NCE`, `SEM_DETACH_STEPS`, ... |
| `sem_targets.py` | Not a plugin: the single target rule (which turn is a report, which text the teacher embeds) plus the sidecar reader and writer; stdlib + numpy only | imported by the SEM plugin and the sidecar builder | none that changes a target (by design) |

## 3. What is NOT in this branch

These files exist in the working tree on `qrnd3` but were left out on purpose, so they are neither versioned nor reviewed here:

| left out | what it is | consequence |
|---|---|---|
| `examples/train/liger/06_*.sh` … `19_*.sh` (incl. `13_…stage3…`, `14_stage5…`, `16_stage6…`, `17_stage6b…`, `18_stage6c_posneg.sh`), `internvl3_5_2b_*.sh`, `qure_full_2b_*.sh`, `16_stage6_toolcalling.sh.run2_frozen_vit` | Earlier-stage launchers | The stage-7 launcher and README refer to `18_stage6c_posneg.sh` for derivations; ask for the file on `qrnd3`. `qure_stage3_plugin.py`'s header still says only `13_…` loads it; the 6b/6c/7 launchers do too. |
| `examples/deploy/*` except `untie_lm_head.py` and `export_lm_only.py` | Eval and serving harnesses (`infer_*`, `run_*`, `score_head_batch.py`, `mld_head_service.py`, `host_*` …) | The SEM canary ([sem_latent README §4.5](examples/train/liger/sem_latent/README.md#45-serve--export)) has not been added to them. |
| `examples/train/training_test_split/` | Includes `split_qwen3_5_2b_cls_json_mldecoder.sh` | `qure_plugin_mode.py` keeps `swift export --to_cached_dataset` in train mode because of this script. |
| `paper/`, `paper_vlm/`, `notebooks/deploy/` | Papers, serving Dockerfile, notebooks | `paper_vlm/plugins/` holds older plugin copies without the F1 dedup. `paper_vlm/serving/docker/Dockerfile.qwen3vl` installs upstream `ms-swift[all]`, so it serves a SEM checkpoint without SEM tokens. |
| `swift_plugins/mldecoder_joint_plugin.py.pre_logfix`, `.pre_rowmask`, every `__pycache__/`, `--help`, `.gitmodules`, the local edit to `qure_full_4b.sh` | Local backups and stray files | |
| `qure_chunked_ce.py` | Referenced by `qure_graded_loss_scale.py`'s header | Not in this repo at all. |
| Data, sidecars, checkpoints | Everything under `/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/` | Never in git ([§9](#9-data-governance)). |

## 4. Environment

| item | value on `qrnd3` |
|---|---|
| conda env | `/home/ubuntu/miniconda3/envs/ms-swift-may-s4` (Python 3.12.13) |
| swift | `ms-swift 4.4.0.dev0`, an **editable** install of this working tree (`import swift` resolves to `/fastdata4/sahil_work/VLM_qure/vlm/swift/swift/`) |
| key packages | torch 2.9.1+cu128, transformers 5.2.0, deepspeed 0.19.0, liger_kernel 0.8.0, flash_attn 2.8.3, sglang 0.5.10, albumentations 1.3.1; **no vLLM** |
| other envs | `ms-swift-may` imports its own site-packages swift, **not** this tree, so it runs none of the fork's core changes. `sg-lang` has sglang 0.5.10 and vLLM 0.20.2, for native OpenAI servers on `-lm` exports. |

```bash
export PATH=/home/ubuntu/miniconda3/envs/ms-swift-may-s4/bin:$PATH
cd /fastdata4/sahil_work/VLM_qure/vlm/swift
export FM_ACCELERATE_REPO=/fastdata4/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate
export PYTHONPATH=$PWD:$FM_ACCELERATE_REPO        # repo root: `import swift_plugins.*`; FM repo: MLDecoder, transform_utils
```

- **`PYTHONPATH` must contain the repo root.** `swift_plugins` is not installed; plugins import `swift_plugins.qure_plugin_mode`, and the SEM plugin and the builder import `swift_plugins.sem_targets`. The editable install covers `swift` only.
  - Keep the root on `PYTHONPATH` even in an env whose swift is not this tree. Otherwise the CLI would silently run another copy of swift core.
- **`FM_ACCELERATE_REPO`** (default above) provides `MLDecoder` (mldecoder_joint, mldecoder_joint_internvl, mldecoder_infer) and `data.utils.transform_utils` (transforms_qure). The launchers export it.
- **Storage roots.** Paths under `/localstorage`, `/fastdata3` and `/fastdata4` are interchangeable: vision_utils, `sem_targets.resolve_storage_path` and the stage-7 launcher try the other roots when a path is missing. On `qrnd3`, `/fastdata4` is an NFS mount of the storage server's `/localstorage` export.
- **New machine.** The env was not exported. The equivalent is an editable install of this checkout (`pip install -e .`) with the versions above, plus the two `PYTHONPATH` entries. Nothing here has been tried on a host other than `qrnd3`.
- **Long GPU jobs.** Start them with `setsid nohup … & disown`, or `DETACH=1` for the stage-7 launcher, so they survive the shell. The launcher also sets `oom_score_adj=1000` on itself and never kills anything.

## 5. Bug fixes F1-F4

All four were pre-existing bugs, found while building stage 7. Each is on by default and has an escape hatch that restores the legacy behaviour. Evidence is in `tests/qure/test_plugin_fixes.py` (25 tests).

| | symptom (before) | fix | escape hatch |
|---|---|---|---|
| **F1** plugin double import | swift imports a plugin as a top-level module (`qure_augment_plugin`); other plugins imported it as `swift_plugins.qure_augment_plugin`. Python ran the file **twice**. Effects:<ul><li>`qure_stage3`'s import stacked a second RandAugment wrapper, so non-bbox images were augmented **twice**. The bbox guard zeroed `AUG_PROB` on the wrong copy, so bbox rows still got **geometric** augmentation. The 6c train log shows `_preprocess_inputs patched` twice.</li><li>`mldecoder_infer` executed a second `mldecoder_joint`, whose train patches were installed twice.</li><li>The full 6c list plus `mldecoder_infer_plugin` crashed at import (`'function' object has no attribute '__func__'`).</li></ul> | Every plugin calls `dedup_module(__name__, '<name>')` as its first statement, so the module is registered under both names and never re-executes. Every patch is `once('<name>.<fn>')`-guarded. `qure_stage3` resolves the augment module that is actually installed. `drop_failed_import` keeps a failing import (the albumentations check) loud. | `QURE_PLUGIN_DEDUP=0`: legacy double import, reproduced byte for byte (30/30 seeded train encodes with plugins, pixel values included). |
| **F2** inference never detected under the `swift` CLI | The `swift` console script strips the verb and re-executes `swift/cli/<verb>.py`, so `qure_plugin_mode` always said `train`. `swift infer` / `deploy` on a checkpoint (whose `args.json` restores the training plugins) installed every train-only patch: augmentation hooks, the 1.1M-key cls-label sidecar load, ASL and optimizer wrappers. | The mode is read from `basename(argv[0])` (`infer.py`, `deploy.py`, `eval.py`, `export.py`, `app.py`, `web_ui.py`, `rollout.py`, `merge_lora.py` → infer). `export --to_cached_dataset true` stays train, matched the way swift's parser matches the flag. | `QURE_PLUGIN_MODE=train` (legacy); `=infer` forces inference. In-process Python harnesses (whose `argv[0]` is not a swift CLI file) are still detected as `train`, as before; set `QURE_PLUGIN_MODE=infer` there. |
| **F3** train/serve `lm_head` mismatch | Checkpoints store a separate, trained `lm_head.weight`, but `config.json` says `tie_word_embeddings: true` at the top level and in `text_config`. HF / swift train and run untied. sglang 0.5.10 and vLLM 0.20.2 read the text-level flag, tie, and **serve `lm_head = embed_tokens`**. A later `resize_token_embeddings()` re-ties and discards the trained head. ckpt-608: max \|lm_head − embed\| 0.00381, mean 3.2e-5 (about 0.3% of mean \|lm_head\|), all 248,320 rows differ. | `qure_untie_config_plugin.py` (opt-in) and the SEM plugin (`SEM_UNTIE_CONFIG=1`) set the flag to false at both levels on the in-memory config, so saved checkpoints are untied. Weights are never touched. For existing checkpoints, `examples/deploy/untie_lm_head.py` (dry run by default). **It has not been applied to any existing checkpoint.** | `SEM_UNTIE_CONFIG=0`, `QURE_UNTIE_CONFIG=0`; `export_lm_only.py --no-untie`; `untie_lm_head.py` backs up to `config.json.tied_bak`. |
| **F4** `-lm` export script missing | `checkpoint-608-lm` existed, but the script that made it was not in the repo. | `examples/deploy/export_lm_only.py <ckpt> <out>`. Drops `ml_decoder_head.*`, `asl_loss.*`, `sem_head.*`, optimizer / rng / trainer state, `global_step*` and `args.json`. Applies the F3 untie and installs the SEM jinja when the tokenizer has SEM tokens. Writes `export_manifest.json`. `--dry-run` on ckpt-608 gives 643 → 618 keys and the same file set as the existing `checkpoint-608-lm`. | `--no-untie`, `--no-sem-template`, `--keep-args-json`, `--drop-prefix` |

**What the fixes visibly change:**
- **F1:** anyone re-running a 6b/6c recipe now gets one augmentation per non-bbox image and photometric-only augmentation on bbox rows. `QURE_PLUGIN_DEDUP=0` restores 6c exactly.
- **F2:** `swift infer` installs no train-only patch. Encodes were verified identical in both modes: input_ids, labels, pixel_values and image_grid_thw hashes on 3 requests, pt and sglang.
- **F3:** a stage-7 checkpoint served by sglang or vLLM uses the trained `lm_head`, whereas 6b/6c were served with `embed_tokens`. Part of any served 6c-vs-7 difference is this fix, not SEM.
  - Measured: the stage-7 smoke checkpoints (`checkpoint-15`, `checkpoint-30`) carry `tie_word_embeddings: false` at both levels.
  - Not yet measured: the effect on actual sglang output (GPU step G6 in the verification plan).

## 6. Backward-compatibility guarantees

The hard requirement for this branch was that any script, checkpoint or tokenizer without SEM tokens encodes byte-identically and trains and infers exactly as before, except for the documented F1-F4 fixes, each of which has an escape hatch.

| guarantee | evidence (CPU, 2026-09-30) |
|---|---|
| Core SEM insertion is a strict no-op without `<|sem_0|>`: the same list objects flow on | `test_sem_insertion.py`: 72/72 records identical to the pre-change `base.py` on every key (input_ids, labels, pixel_values, image_grid_thw, lengths, …). 9 real rows × 8 mode variants (train, pt, sglang, vllm). A deliberately broken `base.py` is flagged in 12 of 72 (comparator sensitivity). |
| Same, re-proved independently | 420/420 encodes identical: 30 random real rows from 9 files × 14 mode variants (train, transformers, sglang, vllm, lmdeploy, and with `token.txt` re-added), `Template.decode` included. Negative controls: train vs `loss_scale default` labels differ on 18/30 rows. |
| Cost of the no-op path | 1.9 µs per `_encode_context_list` call (two O(1) vocab lookups; `len(tokenizer)` is never used) |
| Plugin patch sets unchanged except F1 | The stage-6c list under `sft` argv installs the same 9 patched attributes. The only chain difference is the removed second augment wrapper. The InternVL list, the graded-loss-scale list and the inference plugins are identical, or differ only by F1/F2. |
| Escape hatches are exact | `QURE_PLUGIN_DEDUP=0`: 30/30 seeded train encodes with plugins identical to the pre-change plugins, pixel values included. `QURE_PLUGIN_MODE=train` (with `DEDUP=0`) gives an identical patch set under `infer` argv. |
| F3 does not change training | ckpt-608 loaded through `get_model_processor` with the shipped (tied) and an untied config: `lm_head` and `embed_tokens` are bit-identical, and `lm_head` equals the saved `lm_head.weight`. |
| Nothing existing was rewritten by this branch's SEM work | No launcher (`1*_*.sh`), dataset, checkpoint or `token.txt` edit. The SEM tokens come from a separate `token_sem.txt`. At final verification no `config.json.tied_bak` existed under the output dirs. |
| One behaviour the hatch cannot restore | The pre-change import crash ("6c list + `mldecoder_infer_plugin`") now loads. |

The full record is `review/final_verify.out` from the build session. It is not in the repo; the numbers above are copied from it.

## 7. Tests

Everything is under `tests/qure/`, runs on CPU only (`CUDA_VISIBLE_DEVICES=""`), and needs this host's stage-6b checkpoint-608 and training data. Where those are missing, `conftest.py` leaves the directory out of pytest collection (`QURE_TESTS_FORCE=1` collects anyway). No test writes under the repo, the checkpoint or the datasets.

```bash
export CUDA_VISIBLE_DEVICES="" PYTHONPATH=/fastdata4/sahil_work/VLM_qure/vlm/swift
PY=/home/ubuntu/miniconda3/envs/ms-swift-may-s4/bin/python
T=/fastdata4/sahil_work/VLM_qure/qure_test_out            # any scratch dir; set per file (defaults below)
```

| command | covers | last recorded result | wall | peak RSS |
|---|---|---|---|---|
| `SEM_TEST_OUT=$T/ins SEM_TEST_BACKUP=<see below> $PY tests/qure/test_sem_insertion.py` | SEM insertion: byte identity without SEM, one block per header with SEM, labels -100, mrope, jinja == sglang ids, `SWIFT_SEM_LATENT=0`, thread safety | 53/53 checks (pytest: 7 passed) | ~104 s | 5.9 GB |
| `SEM_TEST_OUT=$T/tgt SEM_TEST_E2E=1 $PY tests/qure/test_sem_targets.py` | The target rule vs the full census; sidecar round-trip; builder resume and provenance; real-teacher CPU end-to-end (`SEM_TEST_E2E=1`) | 20/20 (pytest: 20 passed) | ~34 s | 24.2 GB with E2E |
| `QURE_TEST_OUT=$T/fix $PY tests/qure/test_plugin_fixes.py --slow` | F1-F4, including the real `swift infer/deploy/sft/export` CLI and two full ckpt-608 CPU loads (`--slow`, or `QURE_SLOW=1` under pytest) | 25/25 (pytest: 25 passed) | ~6.7 min | 9.1 GB |
| `SEM_TEST_OUT=$T/plug SEM_TEST_ONLINE=1 $PY tests/qure/test_sem_plugin.py` | SEM plugin: loss math vs float64, head, queue, detach, ZeRO-2 fallback order, optimizer group, refusals, real-template targets, two toy end-to-end scenarios, 2-rank gloo gather, online teacher | 157/157 checks, 30/30 tests | ~168 s | **122.5 GB** |
| `SEM_TEST_OUT=$T/integ $PY tests/qure/test_sem_integration_cpu.py` | ckpt-608 real weights, forward + backward with SEM under reentrant gradient checkpointing, head restore | 11/11 checks (pytest: 1 passed) | ~53 s | 23.9 GB |
| `STAGE7_TEST_OUT=$T/l7 $PY tests/qure/test_stage7_launcher.py` | The stage-7 launcher through a fake `swift`: sidecar refusals, dry-run argv, `DETACH`, guards, final-argv refusals, `MIX` defaults | 14/14 (pytest: 14 passed) | ~214 s | 1.3 GB |
| `for f in tests/qure/test_*.py; do $PY -m pytest -q -p no:cacheprovider $f; done` (one process per file, each with its own `*_TEST_OUT` as above) | All of the above | 97 collected; 97/97 from a clean checkout of this branch | ~7 min when the files run in parallel | max of the above |

**RAM.**
- `test_sem_plugin.py` peaks at about 122 GB because of its two toy end-to-end scenarios (full-vocab fp32 logits, 248,320 × [4, 5026], without Liger). Run it only on a large-RAM host. Its other tests stay under 25 GB. `$PY tests/qure/test_sem_plugin.py -k <substring>` runs a subset.
- The rest fit in about 25 GB each.
- The six files do not interfere with each other and were verified running concurrently. Budget about 190 GB if you do the same.

**Where outputs go.** Do not run all files in ONE pytest process with ONE shared `SEM_TEST_OUT`: `test_sem_insertion.py` and `test_sem_plugin.py` both write `rows.jsonl` there in different formats (29 `KeyError: 'raw'` failures). Each file writes artefacts to its `*_TEST_OUT` directory. The defaults point at the build session's scratch directory (a `/tmp/claude-1000/...` path). Only some files fall back to the system temp dir when that path is missing. Set the variables explicitly, as above.

**The byte-identity test needs the pre-change `base.py`.** `test_sem_insertion.py` (a) compares against a tree with the pre-change `swift/template/base.py` restored. That file is exactly commit 1's version, so rebuild the backup from git:

```bash
C=$(git log --format=%H --diff-filter=A -- swift/template/sem_latent.py | tail -1)   # the commit that added SEM
B=$T/pre_change_backup; mkdir -p $B/swift/template
git show $C~1:swift/template/base.py > $B/swift/template/base.py
SEM_TEST_BACKUP=$B $PY tests/qure/test_sem_insertion.py
```

`test_stage7_launcher.py::test_hook_negative_control` looks for the same file at the build-session path and prints `SKIP` when it is absent.

## 8. Stage-7 quickstart

Stage 7 adds K = 8 `<|sem_i|>` tokens right after every assistant header. A small head (`sem_head.*`) regresses their final hidden states onto the frozen teacher embedding (`bb_23sep`, 2560-d) of that turn's ground-truth report. The start point is stage-6b checkpoint-608. The AR + ASL losses, augmentation, Liger, ZeRO-2 and flash-attention are as in stage 6c; the default data (`MIX=mcr`) is MIMIC/CheXpert + ReXGradient only. The full method and every knob are in [`examples/train/liger/sem_latent/README.md`](examples/train/liger/sem_latent/README.md).

```
 stage-6c files + ReXGradient ---build_stage7_mcr_data.py (CPU)---> stage7_mcr/*.jsonl
                                                                          |
                                     run_build_sem_targets.sh (8 GPUs) <--+
                                                  |
                                                  v
                                     sem_targets/stage7_mcr_rule_v1/  (teacher sidecar; meta.json = complete)
                                                  |
 20_stage7_sem_latent.sh:  DRY_RUN=1 (prechecks) -> DETACH=1 (train) -> logging.jsonl / TensorBoard
                                                  |
                                                  v
                     <run>/checkpoint-N --- swift infer (THIS fork) ---------------> reports
                                        \-- export_lm_only.py -> checkpoint-N-lm --> native sglang / vLLM
```

| step | command | status on 2026-09-30 |
|---|---|---|
| 0. data (`MIX=mcr`, CPU, ~4 min) | `CUDA_VISIBLE_DEVICES="" python examples/train/liger/sem_latent/build_stage7_mcr_data.py` | **Built** (`training_data/stage7_mcr/`, deterministic, leakage checks pass) |
| 1. teacher sidecar (8 GPUs, resumable) | `mkdir -p <OUT> && GPUS="0 1 2 3 4 5 6 7" setsid nohup bash examples/train/liger/sem_latent/run_build_sem_targets.sh > <OUT>/build.log 2>&1 < /dev/null & disown` (default `OUT=…/sem_targets/stage7_mcr_rule_v1`) | `MIX=6c` sidecar built (218,505 targets, about 2 min wall on 8 GPUs). **`MIX=mcr` sidecar not built yet**: only its CPU text collection exists, so the launcher refuses and prints this command. |
| 2. dry run | `DRY_RUN=1 bash examples/train/liger/20_stage7_sem_latent.sh` | Passes with `MIX=6c`; with the default `MIX=mcr` it stops at the missing sidecar. |
| 3. launch | `DETACH=1 bash examples/train/liger/20_stage7_sem_latent.sh` (prechecks in the foreground, then `setsid nohup`; log `$OUTPUT_DIR/train_<date>.log`) | Only smoke runs so far (below). No full run. |
| 4. monitor | `logging.jsonl` / TensorBoard: `train/sem_share`, `train/ar_loss`, `train/sem_cos_loss`, `train/sem_kl`, `val_sem/*`, `val_sem_unseen/*` | See the loss-balance guidance below. |
| 5. eval | `val_sem_unseen/*` (targets not in any train file) and ReX on the public test split only, with the fict prompt; `eval_predictions/sem_step*.npz` for offline analysis | `eval_loss` includes the SEM terms and is not comparable with 6b/6c. |
| 6. serve / export | `swift infer --model <ckpt> --enable_thinking false …` from **this** fork; or `python examples/deploy/export_lm_only.py <ckpt> <ckpt>-lm`, then a native sglang/vLLM server | Not yet run on a SEM checkpoint (GPU steps G4-G6). Any other swift build serves a SEM checkpoint with zero SEM tokens and says nothing; put the canary from the stage-7 README §4.5 in your harness. |

**Measured so far on GPU** (details and the table in [sem_latent README §7](examples/train/liger/sem_latent/README.md#7-validation-so-far)):
- **Teacher.** bf16 vs fp32 cos ≥ 0.9998; centred off-diagonal cos 0.002.
- **A/B runs.** 4 GPUs × 30 steps with ZeRO-2 + Liger + flash-attention, SEM vs no SEM, on MIMIC/CheXpert-only rows. Both completed with no hang. CE was unchanged (0.1700 vs 0.1701) and eval token accuracy was unchanged (0.9529 vs 0.9528).
- **Checkpoints.** They carry `sem_head.*` (7 tensors), `ml_decoder_head.*` (23), the SEM jinja, SEM ids 248211..248218 and untied configs.
- **The head is still at chance** (val centred cos 0.011). The smoke supervised only a few blocks per step.
- **An 8-GPU smoke on the old 6c mixture hung at step 3** (all ranks in NCCL). That mixture had 178 unreadable images replaced by noise. The cause is not proven.

**Loss balance.** In the smoke, the SEM term was about half of the total loss once the InfoNCE queue had entries: SEM term about 0.2-0.3; `train/sem_share` mean 47%, 26-73% after step 10; soft-InfoNCE KL typically 3-5 at `SEM_W_NCE=0.05`. After the `SEM_DETACH_STEPS` warm-up, watch `train/sem_share` and `train/ar_loss`. If the CE rises, lower `SEM_W_NCE` to about 0.02, or schedule it with `SEM_W_NCE_START/_END/_DECAY_STEPS`.

**Known data issue before a real `MIX=mcr` run.** 5,018 B3 `N7` rows teach the `detect` tool to return only empty boxes (`{"boxes": []}`), and no row with a real box survives the MIMIC/CheXpert filter. See [sem_latent README §8 D1](examples/train/liger/sem_latent/README.md#8-known-data-quality-issues-mixmcr) for the options.

## 9. Data governance

- **No data, sidecars or checkpoints in git.** Training data, teacher sidecars, checkpoints and exports live under `/fastdata4/sahil_work/VLM_qure/data_preparation/training_data/` (outside the repo). The teacher and probe live under `/fastdata4/sahil_work/tag_extractor/`. Only code, token lists and docs are committed. `.gitignore` also excludes local inference and analysis dirs. Do not `git add` a `*.jsonl`, `*.npy`, `*.safetensors` or `output/` path.
- **ReXGradient-160K is under a non-commercial research licence.** The ReXGradient-160K Non-Commercial Data Access and Use Agreement (`/fastdata4/sahil_work/ReXGradient-160K/README.md`) allows non-commercial research and non-clinical use only, with no redistribution to third parties. Therefore:
  - `training_data/stage7_mcr/REX_TRAIN.jsonl` / `REX_VAL.jsonl`, and any sidecar built from them (`sem_targets/stage7_mcr_rule_v1`), stay internal. They are never pushed to HF or any other hub, and never committed.
  - A checkpoint trained with `MIX=mcr` (which includes REX_TRAIN) is **research-only**: no commercial or clinical use, and no public release.
  - The ReX TEST split is never used for training; the builder asserts this, and an independent check confirmed 0 overlap. Once a model is trained on REX_TRAIN, evaluate ReX on the public test split only. Existing "all 160k" ReX numbers for such a model are contaminated.
- **MIMIC-CXR and CheXpert** images (the rest of `MIX=mcr`) are covered by their own data-use agreements, which are not restated here.
- **Checkpoints on the Hub.** Any upload of a checkpoint trained on `MIX=mcr` (or any other mixture containing REX_TRAIN) must respect the ReXGradient terms above. Pre-stage-7 checkpoints are unaffected by this branch.
