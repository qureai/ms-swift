CUDA_VISIBLE_DEVICES=0 \
swift deploy \
    --model internlm/Intern-S1-mini \
    --model_type interns1 \
    --template interns1 \
    --torch_dtype bfloat16 \
    --infer_backend sglang \
    --host 0.0.0.0 \
    --port 8000 \
    --max_new_tokens 2048 \
    --temperature 0 \
    --sglang_context_length 32768 \
    --sglang_tp_size 1 \
    --sglang_mem_fraction_static 0.90 \
    --served_model_name interns1_mini

# Client test after the server is up:
# curl http://localhost:8000/v1/chat/completions \
# -H "Content-Type: application/json" \
# -d '{
# "model": "interns1_mini",
# "messages": [{"role": "user", "content": "What is your name?"}],
# "temperature": 0
# }'
