#!/bin/bash
# End-to-end `swift sft` smoke test for the CT template + secondary 3D encoder.
#
# Run this in a GPU environment whose torch matches the CUDA driver. (In swift_2026 the installed
# torch is built for cu130 while the node driver is CUDA 12.05, so the GPUs are unavailable there.)
#
# 11 windows (full_range base + 10 anatomical) == Pillar0-ChestCT's native in_channels (11), so the
# pretrained patch embed is kept as-is. --ct_volume_size 128,128,128 yields Atlas's 3-scale layout.
#
# Photon-style alternative — use a Qwen-VL tower itself as the 3D encoder (auto-detected):
#   --vision_3d_model Qwen/Qwen2.5-VL-3B-Instruct   (no trust_remote_code needed)
#   --ct_windows lung,bone                          (base+2 = 3 chans, matches Qwen's native in=3)
#   --ct_volume_size 16,224,224                     (T%2==0, H,W % (patch*merge=28) == 0)
set -euo pipefail
cd "$(dirname "$0")/.."

python test_cases/make_dummy_ct_data.py --n 4 --shape 160,160,160

USE_HF=1 swift sft \
  --model Qwen/Qwen2.5-VL-3B-Instruct \
  --vision_3d_model YalaLab/Pillar0-ChestCT \
  --vision_3d_trust_remote_code true \
  --vision_3d_max_tokens 16 \
  --template qwen2_5_vl_ct \
  --ct_windows lung,mediastinum,abdomen,liver,bone,brain,subdural,stroke,temporal_bone,soft_tissue \
  --ct_volume_size 128,128,128 \
  --dataset test_cases/dummy_ct_data/ct_dummy.jsonl \
  --tuner_type lora \
  --split_dataset_ratio 0 \
  --max_steps 2 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --torch_dtype bfloat16 \
  --attn_impl flash_attention_2 \
  --use_hf true \
  --dataloader_num_workers 0 \
  --dataset_num_proc 1 \
  --logging_steps 1 \
  --output_dir output/ct_smoke
