# Max-throughput vLLM deployment for InternVL3.5-2B (medical).
# A 2B model fits comfortably on a single GPU, so we run 8 independent
# data-parallel replicas (TP=1) instead of tensor-parallel sharding:
# no cross-GPU communication => highest aggregate throughput.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
VLLM_ATTENTION_BACKEND=FLASHINFER \
swift deploy \
    --model /localstorage/sahil_work/VLM_qure/vlm/swift/output/11_june_InternVL3_5-2B-Instruct_pretraining_cleaned_bbox_segmed_prod/v16-20260614-093456/last \
    --torch_dtype bfloat16 \
    --infer_backend vllm \
    --host 0.0.0.0 \
    --port 8000 \
    --max_new_tokens 16000 \
    --temperature 0.7 \
    --vllm_max_model_len 25000 \
    --vllm_tensor_parallel_size 1 \
    --vllm_pipeline_parallel_size 1 \
    --vllm_data_parallel_size 8 \
    --vllm_gpu_memory_utilization 0.90 \
    --vllm_max_num_seqs 512 \
    --vllm_enable_prefix_caching true \
    --vllm_enforce_eager false \
    --vllm_disable_custom_all_reduce true \
    --vllm_limit_mm_per_prompt '{"image": 6}' \
    --vllm_engine_kwargs '{"max_num_batched_tokens": 8192}' \
    --served_model_name internvl3_5_2b_medical

# Long-context version, if needed:
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# swift deploy \
#     --model /localstorage/sahil_work/VLM_qure/vlm/swift/output/11_june_InternVL3_5-2B-Instruct_pretraining_cleaned_bbox_segmed_prod/v16-20260614-093456/last \
#     --model_type internvl_chat \
#     --template internvl3_5 \
#     --system "You are a helpful medical assistant." \
#     --torch_dtype bfloat16 \
#     --infer_backend vllm \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --max_new_tokens 15000 \
#     --temperature 0 \
#     --vllm_max_model_len 28000 \
#     --vllm_tensor_parallel_size 1 \
#     --vllm_pipeline_parallel_size 1 \
#     --vllm_data_parallel_size 8 \
#     --vllm_gpu_memory_utilization 0.85 \
#     --vllm_max_num_seqs 256 \
#     --vllm_enable_prefix_caching true \
#     --vllm_enforce_eager false \
#     --vllm_disable_custom_all_reduce true \
#     --vllm_limit_mm_per_prompt '{"image": 6}' \
#     --vllm_engine_kwargs '{"max_num_batched_tokens": 8192}' \
#     --served_model_name internvl3_5_2b_medical
