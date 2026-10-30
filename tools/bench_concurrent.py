#!/usr/bin/env python3
"""演示连续批处理效果：N 个客户端同时请求 vs 串行（只在 B>1 的服务上生效）。

用法（宿主）：python3 tools/bench_concurrent.py [N]
"""
import json
import sys
import threading
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 2
QS = ["用一句话说明什么是张量", "用一句话说明什么是梯度", "用一句话说明什么是矩阵",
      "用一句话说明什么是向量"]


def one(i, out):
    q = QS[i % len(QS)]
    body = json.dumps({"messages": [{"role": "user", "content": q}],
                       "max_tokens": 32, "temperature": 0}).encode()
    r = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    d = json.load(urllib.request.urlopen(r, timeout=900))
    out[i] = (time.perf_counter() - t, d["choices"][0]["message"]["content"])


# 串行基线：逐个发
t0 = time.perf_counter()
ser = {}
for i in range(N):
    one(i, ser)
t_ser = time.perf_counter() - t0

# 并发
t0 = time.perf_counter()
par = {}
ths = [threading.Thread(target=one, args=(i, par)) for i in range(N)]
for th in ths:
    th.start()
for th in ths:
    th.join()
t_par = time.perf_counter() - t0

print(f"  N={N} 请求，每个 max_tokens=32")
print(f"  串行总墙钟 {t_ser:6.2f}s  （平均 {t_ser/N:5.2f}s/请求）")
print(f"  并发总墙钟 {t_par:6.2f}s  （平均 {t_par/N:5.2f}s/请求）")
print(f"  ⇒ 吞吐提升 {t_ser/t_par:4.2f}×")
for i in range(N):
    print(f"    #{i}: 串行 {ser[i][0]:5.2f}s / 并发 {par[i][0]:5.2f}s  "
          f"{par[i][1][:34]!r}")
