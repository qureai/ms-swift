# 3D vision encoder — test cases

Tests for the optional secondary 3D vision encoder (`--vision_3d_model`), which loads a vision
encoder from any model name and uses it as a CT-volume backbone alongside the VLM's built-in 2D
encoder. Volumes ride the existing `<video>` token (no new special token).

## Environment

Use `swift_2026` (HuggingFace login lives there). Note the env sits on a NAS, so the **first**
`import swift` is slow (~10 min, cold file cache); each test process imports once and runs all cases.

Extra deps installed for these tests: `einops timm nibabel` (Atlas + CT IO), plus swift runtime deps.

```bash
ENV=/cache/fast_data_nas8/janhavi/miniconda3_v2/envs/swift_2026
cd /cache/fast_data_nas8/janhavi/VLM_qure/vlm/swift
```

## 1. Model loading — `test_vision_3d_loading.py`

Validates load → extract vision tower → Conv2d→Conv3d (if needed) → wrap in `Vision3DBackbone`.

```bash
# fast: Conv2d->Conv3d unit checks only (no downloads)
$ENV/bin/python test_cases/test_vision_3d_loading.py --quick

# Atlas / Pillar0 (natively 3D, custom code, trust_remote_code)
USE_HF=1 $ENV/bin/python test_cases/test_vision_3d_loading.py --models atlas

# A standard VLM whose 2D vision tower gets inflated to 3D
USE_HF=1 $ENV/bin/python test_cases/test_vision_3d_loading.py --models qwen,internvl

# multi-channel windowing -> in_channels = 4
USE_HF=1 $ENV/bin/python test_cases/test_vision_3d_loading.py --models atlas --in-channels 4
```

Defaults: `atlas=YalaLab/Pillar0-ChestCT`, `qwen=Qwen/Qwen2.5-VL-3B-Instruct`,
`internvl=OpenGVLab/InternVL3-1B-hf`. Override with `--atlas-model/--qwen-model/--internvl-model`.
Each model case is isolated — one failure does not abort the others.

## 2. Dummy CT data — `make_dummy_ct_data.py`

No real CT dataset path exists yet, so this fabricates synthetic `.nii.gz` volumes + a swift JSONL
(`<video>` → volume path) for dataset-loading / `swift sft` smoke tests.

```bash
$ENV/bin/python test_cases/make_dummy_ct_data.py --n 4 --out test_cases/dummy_ct_data
```

## 3. Dataset loading / `swift sft` smoke (added in Step 2)

Will exercise: 2D data at `/cache/fast_data_nas8/vlm_team_data/janhavi/11_Nov_pretraining_data_without_bbox.json`
and the dummy CT JSONL above, through the CT template + dual (2D/3D) routing.
