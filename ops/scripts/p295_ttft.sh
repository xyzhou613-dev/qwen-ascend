#!/bin/bash
# ★ 冷/热 TTFT 对照（用随机串保证不命中缓存）
set -u
python3 - <<'PY'
import json, time, urllib.request, uuid
URL = "http://127.0.0.1:8000/v1/chat/completions"

def gen(prompt, mx=8, label=""):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": mx, "temperature": 0, "stream": True}).encode()
    r = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; k = 0
    for line in urllib.request.urlopen(r, timeout=900):
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
    prof = json.load(urllib.request.urlopen("http://127.0.0.1:8000/fnx/prof", timeout=30))["last_prof"]
    print(f"  {label:22s} prompt={prof['n_prompt']:3d} hit={prof['prefix_hit']:3d} "
          f"prefill={prof['n_prefill']:3d}  TTFT={first:6.2f}s 总={tot:5.2f}s 出={k}tok")

# 冷：随机串唯一，不命中任何缓存
u = uuid.uuid4().hex[:6]
gen(f"你好，请回一个字。编号 {u}", 8, "冷 · 极短")
gen("请记住下面这串编号并只回复 OK：" + uuid.uuid4().hex, 8, "冷 · 短")
gen(("以下是技术说明：" + uuid.uuid4().hex + "，") * 8 + "请只回复 OK", 8, "冷 · 中(~100 tok)")
gen(("以下是技术说明：" + uuid.uuid4().hex + "，") * 24 + "请只回复 OK", 8, "冷 · 长(~300 tok)")

# 热：随后再问一次完全相同的（前缀+对话缓存命中）
gen(f"你好，请回一个字。编号 {u}", 8, "热 · 同一 prompt")

print()
# 稳态 TPOT（长输出）
for mx in (64, 160):
    body = json.dumps({"messages": [{"role": "user", "content": "请写一段关于秋天的散文"}],
                       "max_tokens": mx, "temperature": 0, "stream": True}).encode()
    r = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; k = 0
    for line in urllib.request.urlopen(r, timeout=900):
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
    tpot = (tot - first) / max(1, k - 1)
    print(f"  稳态输出 max_new={mx:3d}: 出={k:3d}tok 总={tot:6.2f}s "
          f"TPOT={tpot*1e3:6.1f} ms  ⇒ {1/tpot:4.1f} tok/s")
PY
