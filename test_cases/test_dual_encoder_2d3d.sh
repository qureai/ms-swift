#!/bin/bash
# Dual-encoder (2D + 3D) validation run: Qwen3-VL-2B-Instruct as the 2D/host VLM, Pillar0/Atlas as the
# 3D encoder. Trains on a mixed dataset (image-only, volume-only, both) with VISION_3D_DEBUG=1, which
# logs — from the actual forward pass — each vision encoder's output shape and image/video token
# routing per batch. With batch_size=1 and no shuffle, steps 1/2/3 hit each routing case in turn.
#
# 11 windows (full_range base + 10 anatomical) == Atlas native in_channels (11) -> pretrained patch
# embed kept. --ct_volume_size 128,128,128 yields Atlas's 3-scale layout.
#
# Run in a GPU env whose torch matches the driver (e.g. swift_july). If ms-swift is editable-installed
# and you hit a user-site numpy/pandas ABI clash, prefix with PYTHONNOUSERSITE=1.
#
# NOTE: uses --attn_impl sdpa. flash_attention_2 currently errors in this env with a swift/
# transformers-5.12/flash-attn kwarg clash ("flash_attention_forward() got multiple values for
# 'cu_seq_lens_q'") in the attention path — unrelated to the 3D-encoder code; sdpa runs clean.
set -euo pipefail
cd "$(dirname "$0")/.."

python test_cases/make_dummy_mixed_data.py --out test_cases/dummy_mixed_data

export VISION_3D_DEBUG=1 USE_HF=1 TOKENIZERS_PARALLELISM=false
swift sft \
  --model Qwen/Qwen3-VL-2B-Instruct \
  --vision_3d_model YalaLab/Pillar0-ChestCT \
  --vision_3d_trust_remote_code true \
  --vision_3d_max_tokens 16 \
  --template qwen3_vl_ct \
  --ct_windows lung,mediastinum,abdomen,liver,bone,brain,subdural,stroke,temporal_bone,soft_tissue \
  --ct_volume_size 128,128,128 \
  --dataset test_cases/dummy_mixed_data/ct_mixed.jsonl \
  --tuner_type full \
  --split_dataset_ratio 0 \
  --max_steps 3 \
  --per_device_train_batch_size 1 \
  --dataset_shuffle false \
  --dataloader_num_workers 0 \
  --dataset_num_proc 1 \
  --torch_dtype bfloat16 \
  --attn_impl sdpa \
  --use_hf true \
  --logging_steps 1 \
  --report_to tensorboard \
  --output_dir output/dual_encoder_smoke
