IMAGE_MAX_TOKEN_NUM=2500 \
VIDEO_MAX_TOKEN_NUM=128 \
FPS_MAX_FRAMES=16 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift deploy \
    --model Qwen/Qwen3.6-27B \
    --served_model_name Qwen3.6-27B \
    --host 0.0.0.0 \
    --port 8000 \
    --enable_thinking false \
    --use_hf true \
    --infer_backend sglang \
    --sglang_tp_size 8 \
    --max_new_tokens 24000 \
    --sglang_context_length 64768 \
    --sglang_mem_fraction_static 0.9

