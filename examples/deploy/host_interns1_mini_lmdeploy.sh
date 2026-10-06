# Host Intern-S1-mini with lmdeploy's native OpenAI-compatible server.
# swift's lmdeploy backend does not support VLMs on lmdeploy>=0.9, but lmdeploy
# 0.14.0 supports InternS1ForConditionalGeneration natively via its pytorch backend.

USE_HF=1 \
CUDA_VISIBLE_DEVICES=0 \
lmdeploy serve api_server \
    internlm/Intern-S1-mini \
    --backend pytorch \
    --trust-remote-code \
    --dtype bfloat16 \
    --server-name 0.0.0.0 \
    --server-port 8000 \
    --tp 1 \
    --session-len 32768 \
    --cache-max-entry-count 0.9 \
    --max-batch-size 256 \
    --vision-max-batch-size 8 \
    --model-name interns1_mini

# Client test after the server is up:
# curl http://localhost:8000/v1/chat/completions \
# -H "Content-Type: application/json" \
# -d '{
# "model": "interns1_mini",
# "messages": [{"role": "user", "content": "What is your name?"}],
# "temperature": 0
# }'
