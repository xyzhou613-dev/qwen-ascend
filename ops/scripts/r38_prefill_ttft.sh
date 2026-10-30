#!/bin/bash
# p328: prefill/TTFT 与 decode 口径（★必须读满 [DONE]：中途断开会让服务端卡死）
set -u
python3 - <<'PY'
import json, time, urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"


def gen(msg, mx=8):
    b = json.dumps({"messages": [{"role": "user", "content": msg}],
                    "max_tokens": mx, "temperature": 0, "stream": True}).encode()
    r = urllib.request.Request(URL, data=b, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    k = 0
    for line in urllib.request.urlopen(r, timeout=1800):
        s = line.decode("utf-8", "ignore").strip()
        if not s.startswith("data: "):
            continue
        p = s[6:]
        if p == "[DONE]":
            break
        d = json.loads(p)
        if d["choices"][0]["delta"].get("content"):
            k += 1
            if first is None:
                first = time.perf_counter() - t0
    tot = time.perf_counter() - t0
    prof = json.load(urllib.request.urlopen("http://127.0.0.1:8000/fnx/prof", timeout=60))["last_prof"]
    return first, tot, k, prof


def row(tag, msg, mx=8):
    first, tot, k, prof = gen(msg, mx)
    npf = prof.get("n_prefill") or 0
    per = (first * 1e3 / npf) if npf else 0.0
    print("  %-9s prompt=%3d hit=%3d prefill=%3d TTFT=%6.2fs  %.1f ms/prefill-token  out=%d"
          % (tag, prof.get("n_prompt") or 0, prof.get("prefix_hit") or 0, npf, first, per, k), flush=True)


base = "昇腾处理器面向大规模深度学习训练与推理场景。"
row("短", "你好")
row("中", "请总结：" + base * 6)          # ~270 字
row("长", "请总结：" + base * 20)         # ~900 字
row("短#2", "你好")                       # 应命中前缀/对话缓存
PY
