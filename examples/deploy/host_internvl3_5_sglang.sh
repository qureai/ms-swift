CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
swift deploy \
    --model /localstorage/sahil_work/VLM_qure/vlm/swift/output/11_june_InternVL3_5-2B-Instruct_pretraining_cleaned_bbox_segmed_prod/v16-20260614-093456/last \
    --model_type internvl_chat \
    --template internvl3_5 \
    --system "You are a helpful medical assistant." \
    --torch_dtype bfloat16 \
    --infer_backend sglang \
    --host 0.0.0.0 \
    --port 8000 \
    --max_new_tokens 2048 \
    --temperature 0 \
    --sglang_context_length 25000 \
    --sglang_tp_size 1 \
    --sglang_pp_size 1 \
    --sglang_dp_size 8 \
    --sglang_mem_fraction_static 0.90 \
    --sglang_disable_custom_all_reduce true \
    --served_model_name internvl3_5_2b_medical

# Long-context version, if needed:
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# swift deploy \
#     --model /localstorage/sahil_work/VLM_qure/vlm/swift/output/11_june_InternVL3_5-2B-Instruct_pretraining_cleaned_bbox_segmed_prod/v16-20260614-093456/last \
#     --model_type internvl_chat \
#     --template internvl3_5 \
#     --system "You are a helpful medical assistant." \
#     --torch_dtype bfloat16 \
#     --infer_backend sglang \
#     --host 0.0.0.0 \
#     --port 8000 \
#     --max_new_tokens 15000 \
#     --temperature 0 \
#     --sglang_context_length 28000 \
#     --sglang_tp_size 1 \
#     --sglang_pp_size 1 \
#     --sglang_dp_size 8 \
#     --sglang_mem_fraction_static 0.85 \
#     --sglang_disable_custom_all_reduce true \
#     --served_model_name internvl3_5_2b_medical
