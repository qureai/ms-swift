# swift_plugins/: Qure plugins for ms-swift

These are loaded with `swift sft|infer … --external_plugins <file> …`. The repo-level overview (environment, bug fixes F1-F4, tests) is in [`../QURE_README.md`](../QURE_README.md). Every file's header docstring is the authoritative description of its knobs.

**Prerequisites:**
- **Repo root on `PYTHONPATH`.** There is no `__init__.py`; plugins import `swift_plugins.qure_plugin_mode` / `swift_plugins.sem_targets`.
- **`FM_ACCELERATE_REPO`**, for the ML-Decoder class and the augmentation transforms. The default is `/fastdata4/sahil_work/VLM_qure/vision_foundation_2d/foundation_model_accelerate`.

## Plugins

"Mode" says what happens under `swift infer/deploy/eval/export` (see the Mode (F2) bullet under [Conventions](#conventions-for-plugin-code-f1--f2)): **train-only** means the plugin installs nothing at inference; **not gated** means it installs in every mode.

| file | purpose | load it for | key env (defaults) | mode |
|---|---|---|---|---|
| `mldecoder_joint_plugin.py` | ML-Decoder head (132 tags) on the Qwen3.5 vision tower (pre- + post-merger taps). ASL is folded into the model's own loss next to the AR CE, so Liger's fused CE stays on. Labels come from `cls_label_sidecar`. Asserts packing is off. | Qwen3.5 joint training (6b, 6c, 7), **first** in the list | `AR_WEIGHT` (1.0); `ASL_WEIGHT` (set = constant, bypasses the schedule) or `ASL_WEIGHT_START/_END/_DECAY_STEPS/_DECAY_SHAPE` (1.0 → 0.03); `MLD_HEAD_LR` (5e-4); `MLD_HEAD_DTYPE`; `MLD_NUM_OF_GROUPS`; `MLD_DECODER_EMBEDDING`; `MLD_EVAL_PRED_DIR`; `MLD_LOG_EVERY`; `CLS_LABEL_SIDECAR`; `FM_ACCELERATE_REPO` | train-only; the head class stays importable |
| `mldecoder_joint_internvl_plugin.py` | The same head and loss for InternVL3.5 (taps `mlp1`, regroups tiles per image) | InternVL joint training | as above, plus `MLD_HEAD_MAX_TOKENS` | train-only |
| `mldecoder_infer_plugin.py` | Re-attaches the trained head at inference; writes `<result_path>.cls_logits.jsonl` (132 logits / probs / top-k per image) | `swift infer --infer_backend pt` | `MLD_CLS_OUT`, `MLD_TOP_K` (15), `MLD_HEAD_DTYPE` | not gated (inference-only) |
| `qure_augment_plugin.py` | Foundation RandAugment (`swift/template/transforms_qure.py`) on each training image, in `Qwen3_5Template._preprocess_inputs`. Refuses to load without albumentations 1.x (train only). | Qwen3.5 training | `QURE_AUG_PROB` (1.0), `QURE_AUG_CROP_MIN/MAX` (0.10 / 0.15), `QURE_AUG_IM_SIZE` (unset), `QURE_AUG_PHOTO_PROB` (1.0) | train-only |
| `qure_augment_internvl_plugin.py` | The same pipeline for InternVL (`Internvl2Template._preprocess_inputs`, before tiling) | InternVL training | `QURE_AUG_*` | train-only |
| `qure_ce_weight_plugin.py` | Token-id-weighted AR CE through Liger's fused `ce_weight`, with no logits tensor | training with `--use_liger_kernel true` | `QURE_W_TRUE` (6.0), `QURE_W_FALSE` (1.0), `QURE_W_FORMAT` (0.10; the stage 6c/7 launchers set 0.5), `QURE_W_DEFAULT` (1.0), `QURE_W_COORD` (1.0; the launchers set 2.0) | train-only |
| `qure_graded_loss_scale.py` | Registers the non-binary `--loss_scale qure_graded` (per-segment weights for cls-JSON / ASP / graph turns). Its header points at a `qure_chunked_ce.py` that is not in this repo. | runs that pass `--loss_scale qure_graded` | `QURE_W_FORMAT` (0.10), `QURE_W_NAME` (0.30), `QURE_W_FALSE`, `QURE_W_TRUE`, `QURE_W_ASP_FINDING`, `QURE_W_GRAPH_FINDING`, `QURE_W_DEFAULT` | not gated (it only registers a name). **Refused by the SEM plugin**, which needs a binary loss scale. |
| `qure_stage3_plugin.py` | Wrapper add-on: (1) restore `ml_decoder_head.*` from `--model` (HF drops them as unexpected keys); (2) optionally freeze the head; (3) bbox-safe augmentation, i.e. rows mentioning `bbox_2d` get photometric-only aug or none | 6b, 6c and 7 training (its header still says only the stage-3 launcher uses it) | `MLD_RESTORE_HEAD` (1), `MLD_FREEZE_HEAD` (0), `QURE_AUG_SKIP_BBOX` (1), `QURE_AUG_BBOX_MODE` (`photometric` or `skip`) | train-only |
| `force_presence_plugin.py` | Forces each `"presence":` value by a per-finding threshold on the LM's own P(true) | `swift infer --infer_backend pt --max_batch_size 1` | `MLD_FORCE_TAU` (0.5), `MLD_FORCE_TAU_JSON`, `MLD_FORCE_OUT` | not gated (inference-only) |
| `record_presence_plugin.py` | Records P(true) at every `"presence":` slot during normal greedy generation, without changing logits | `swift infer --infer_backend pt`, with `--val_dataset_shuffle false` | `MLD_REC_OUT` | not gated (inference-only) |
| `qure_untie_config_plugin.py` | F3: at `prepare_model`, if `lm_head` and `embed_tokens` are distinct tensors, set `tie_word_embeddings=False` at both config levels so saved checkpoints serve the trained head. Weights are untouched. | opt-in for any launcher; stage 7 does not need it (the SEM plugin does the same with `SEM_UNTIE_CONFIG=1`) | `QURE_UNTIE_CONFIG` (1; 0 = load but install nothing) | train-only |
| `sem_latent_plugin.py` | Stage-7 SEM supervision: per-turn teacher targets in encode, collator and `_post_encode` plumbing, `sem_head`, loss hooks, optimizer group, SEM-row init, head restore, config untie, SEM jinja install, `train/sem_*` and `val_sem/*` logging | stage-7 training, **last** in the list | `SEM_TARGETS_DIR` (required), `SEM_W_COS` / `SEM_W_NCE` (0.05 each), `SEM_DETACH_STEPS` (20), `SEM_QUEUE` (16384), `SEM_HEAD_LR` (2e-4), … (the full list is in [the stage-7 README §5.2](../examples/train/liger/sem_latent/README.md#52-sem-plugin-swift_pluginssem_latent_pluginpy-the-header-has-the-full-semantics)) | train-only. SEM insertion itself is in swift core and needs no plugin. |
| `qure_plugin_mode.py` | **Not listed as a plugin.** Train/infer detection (`is_inference()`, F2) and `dedup_module()` / `once()` / `drop_failed_import()` (F1) | imported by every plugin | `QURE_PLUGIN_MODE` (`train` / `infer` override), `QURE_PLUGIN_DEDUP` (1; 0 = legacy double import) | — |
| `sem_targets.py` | **Not a plugin.** The single target rule for stage 7, plus `SidecarReader` / `write_sidecar`. Stdlib + numpy only; shared byte for byte by the plugin and `examples/train/liger/sem_latent/build_sem_targets.py`; `RULE_VERSION` is checked against the sidecar. | imported by `sem_latent_plugin.py`, the builder and the launcher precheck | `SEM_TEACHER_PATH`, `SEM_PROBE_HEAD` (paths only; nothing changes a target) | — |

## Load-order rules

swift imports each `--external_plugins` file **in list order**, as a **top-level module** (the file's directory goes on `sys.path`, then `import <basename>`). Patches chain on whatever is already installed, so order matters.

1. **Training, Qwen3.5 (the stage 6b / 6c / 7 order):**
   ```
   mldecoder_joint_plugin -> qure_augment_plugin -> qure_ce_weight_plugin -> qure_stage3_plugin
       -> [EXTRA_PLUGINS] -> sem_latent_plugin (stage 7 only, always LAST)
   ```
   - `qure_stage3_plugin` wraps patches that `mldecoder_joint` and `qure_augment` install, so it comes after both.
   - `sem_latent_plugin` must be last because:
     - its loss hook has to run after mldecoder's, so mldecoder's logged `ar` excludes the SEM term;
     - its encode, collator and `_post_encode` wrappers chain on everything before it.
2. **`qure_untie_config_plugin`: list it last** in a non-SEM launcher. Every `swift_plugins/` `prepare_model` wrapper uses the `getattr(TunerMixin.prepare_model, '__func__', TunerMixin.prepare_model)` pattern and tolerates it anywhere. A third-party plugin that reads `TunerMixin.prepare_model.__func__` directly must come **before** it.
3. **Inference plugins** (`mldecoder_infer`, `force_presence`, `record_presence`) go on `swift infer --external_plugins` and need `--infer_backend pt`: sglang and vLLM re-implement the model, so the forward hooks and logits processors never fire there. `mldecoder_infer` may be combined with the joint plugin; it borrows the head class from the same module object (F1).
4. **`args.json` restores the training plugin list** at `swift infer` time (`--load_args` defaults to true). So every training plugin is imported again when you serve a checkpoint. With F2 it recognises inference and installs nothing train-only.
5. **Stage 7 refuses some combinations** at `prepare_model`, and the launcher precheck refuses them on the final argv:
   - `--packing`, `--padding_free` and sequence parallel;
   - `--template_backend jinja`;
   - a `--loss_scale` whose base strategy is `all`, or any non-binary one (e.g. `qure_graded`);
   - `--loss_type`, label smoothing, dft loss and channel loss.

## Conventions for plugin code (F1 / F2)

Every `*_plugin.py` here, plus `qure_graded_loss_scale.py`, follows the same skeleton. `tests/qure/test_plugin_fixes.py::test_f1_every_plugin_dedups_first` enforces the first part statically for every `swift_plugins/*_plugin.py`, so a new plugin must follow it too.

```python
"""<long header: what, how to load, every env knob, mode, escape hatches>"""
# F1 -- must stay the first statements of the module (qure_plugin_mode.dedup_module)
try:
    from swift_plugins.qure_plugin_mode import dedup_module as _dedup_module, once as _once
except ImportError:  # repo root not on sys.path; swift put this plugin's own dir there
    from qure_plugin_mode import dedup_module as _dedup_module, once as _once
_dedup_module(__name__, '<my_plugin>')          # canonical name == file basename

...

def patch_something():
    if not _once('<my_plugin>.patch_something'):   # one install per process, whatever the import name
        return
    orig = getattr(TunerMixin.prepare_model, '__func__', TunerMixin.prepare_model)   # chain, never replace
    ...

from swift_plugins.qure_plugin_mode import is_inference  # noqa: E402
if is_inference():
    logger.info('[my-plugin] inference mode -- train-only patches NOT installed')
else:
    patch_something()
```

- **Dedup (F1).**
  - `dedup_module(__name__, '<name>')` registers the executing module under both `<name>` and `swift_plugins.<name>`, so a later import under the other name returns the same object and nothing re-executes.
  - `once('<name>.<fn>')` is a process-global sentinel (kept in `sys.modules['_qure_plugin_registry']`); a second copy can never stack a second wrapper.
  - If an importer catches an import error and retries, it calls `drop_failed_import('<name>')` first, so a plugin that raises at load (e.g. `qure_augment`'s albumentations check) stays loud.
  - To import another plugin, use `import swift_plugins.<name>`. Better, look up the copy that is actually installed in `sys.modules`, as `qure_stage3_plugin` does for the augment module.
- **Escape hatch.** `QURE_PLUGIN_DEDUP=0` disables all of this (no aliasing; `once()` is always True), which reproduces the legacy double import exactly.
- **Mode (F2).** `is_inference()` is decided once, at import. First match wins:
  1. `QURE_PLUGIN_MODE=train|infer`;
  2. `basename(sys.argv[0])` when it is `swift/cli/<verb>.py` (`infer`, `deploy`, `eval`, `export`, `app`, `web_ui`, `rollout`, `merge_lora` → infer; `sft`, `pt`, `rlhf`, `sample` → train; `export --to_cached_dataset true` → train);
  3. a verb in `sys.argv[1:]`;
  4. `train`.

  Two consequences:
  - An in-process Python harness is detected as `train` unless it sets `QURE_PLUGIN_MODE=infer`.
  - `is_inference()` decides whether a patch is **installed**; `Template.is_training` still decides per batch whether it **acts**. Keep both checks.
- **Style.** Use `logger = get_logger()`, `[tag]`-prefixed log lines, env-var knobs with documented defaults (read at import), and a long header documenting every knob and its escape hatch.
