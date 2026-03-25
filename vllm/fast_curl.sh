curl http://10.140.37.10:8100/v1/chat/completions   \
    -H "Content-Type: application/json"   \
    -d '{
        "model": "Qwen3.5-27B",
        "messages": [{"role":"user","content":"用一句话介绍你自己，不超过256个token"}],
        "temperature": 0.0,
        "max_tokens": 256
    }'