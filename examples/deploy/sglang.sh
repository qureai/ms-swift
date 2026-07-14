# MAX_PIXELS caps the vision-token budget per image server-side (match training).
# The client (infer/sglang/sglang_api_infer.py --max-pixels) can cap further; the
# smaller of the two wins.

MAX_PIXELS=$((3500 * 3500)) \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift deploy \
    --model /cache/fast_data_nas71/janhavi/vlm_ckpts/sglang_symlink_ckpts/SAHIL_Qwen_2_5_vl_8b_2611_corrected_pretraining_cleaned_bbox_segmed_prod/v5-20251201-063754_checkpoint-14422 \
    --infer_backend sglang \
    --max_new_tokens 12000 \
    --sglang_context_length 32000 \
    --sglang_tp_size 4 \
    --served_model_name Qwen2.5_8B

# MAX_PIXELS=$((1500 * 1500)) \
# CUDA_VISIBLE_DEVICES=4,5,6,7 \
# swift deploy \
#     --model /cache/fast_data_nas71/janhavi/vlm_ckpts/sglang_symlink_ckpts/SAHIL_Qwen_2_5_vl_3b_2611_corrected_pretraining_cleaned_bbox_segmed_prod/v2-20251203-004541_checkpoint-8241 \
#     --infer_backend sglang \
#     --max_new_tokens 8000 \
#     --sglang_context_length 32000 \
#     --sglang_tp_size 4 \
#     --served_model_name Qwen2.5_3B

# After the server-side deployment above is successful, use the command below to perform a client call test.

# curl http://localhost:8000/v1/chat/completions \
# -H "Content-Type: application/json" \
# -d '{
# "model": "Qwen3-8B",
# "messages": [{"role": "user", "content": "What is your name?"}],
# "temperature": 0
# }'
