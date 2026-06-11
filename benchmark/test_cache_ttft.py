# 이걸 바로 실행해봐: benchmark/test_cache_ttft.py
import time, requests, json

URL = "http://localhost:8000/v1/chat/completions"
MODEL = "Qwen/Qwen3-14B"
PREFIX = "test " * 2000  # ~2000 tokens (방금 쓴 것과 같은 내용)

def ttft(messages):
    t0 = time.time()
    r = requests.post(URL, json={
        "model": MODEL, "messages": messages,
        "max_tokens": 5, "stream": True,
        "temperature": 0,
    }, stream=True, timeout=120)
    for line in r.iter_lines():
        if line and line != b"data: [DONE]":
            raw = line.decode().lstrip("data: ").strip()
            if raw:
                return time.time() - t0
    return None

msg1 = [{"role": "user", "content": PREFIX}]
msg2 = msg1 + [
    {"role": "assistant", "content": "ok"},
    {"role": "user", "content": "continue"},
]

print("Turn 1 (cold, full recompute):"); t1 = ttft(msg1); print(f"  TTFT = {t1:.3f}s")
print("Turn 2 (same prefix, cache hit?):"); t2 = ttft(msg2); print(f"  TTFT = {t2:.3f}s")
print(f"  비율 t2/t1 = {t2/t1:.2f}  (1.0 = cache 없음, <0.3 = L1 hit)")