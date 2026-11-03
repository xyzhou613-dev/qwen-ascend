#!/usr/bin/env python3
"""peak_bench.py — 实测这台机器（2×910B4）的 FP16 / BF16 峰值算力。

分三步：
  1) 单卡（npu:0 / npu:1）分别测 fp16、bf16 matmul 在 2048 / 4096 / 8192 的 TFLOPS；
  2) 双卡并发（两张卡同时各跑一个大 matmul），测整机聚合 TFLOPS（看是否线性扩展）；
  3) 给出去掉固定开销后的"峰值"估计（最大尺寸下的最好值）。

用法（容器内、独占卡）：
  /root/ascend-dev/venv/bin/python peak_bench.py
"""
import threading
import time

import torch
import torch_npu  # noqa: F401


def bench(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n


def one_card(dev, dtype, n, reps):
    with torch.npu.device(dev):
        a = torch.randn(n, n, device=dev, dtype=dtype)
        b = torch.randn(n, n, device=dev, dtype=dtype)
        ms = bench(lambda: torch.matmul(a, b), n=reps, warmup=1) * 1e3
        del a, b
        torch.npu.empty_cache()
    return ms, 2 * n ** 3 / (ms / 1e3) / 1e12


def main():
    cards = [0, 1]
    print(f"卡数 = {torch.npu.device_count()}   设备名 = {torch.npu.get_device_name(0)}")
    print()

    dtypes = [("fp16", torch.float16), ("bf16", torch.bfloat16)]
    best = {}
    for name, dt in dtypes:
        print(f"=== 单卡 {name.upper()} matmul ===")
        print(f"{'card':>5} {'N':>6} {'ms':>9} {'TFLOPS':>9}")
        for dev in cards:
            for n in (2048, 4096, 8192):
                reps = 3 if n >= 8192 else 5
                ms, tf = one_card(dev, dt, n, reps)
                best[name] = max(best.get(name, 0.0), tf)
                print(f"{dev:>5} {n:>6} {ms:>9.2f} {tf:>9.1f}")
        print()

    # ---------- 双卡并发 ----------
    print("=== 双卡并发（两张卡同时各跑一个 8192³ matmul）===")
    for name, dt in dtypes:
        n = 8192
        mats = []
        for dev in cards:
            with torch.npu.device(dev):
                mats.append((torch.randn(n, n, device=dev, dtype=dt),
                             torch.randn(n, n, device=dev, dtype=dt)))
        torch.npu.synchronize()

        results = [None] * len(cards)

        def worker(i, dev):
            with torch.npu.device(dev):
                a, b = mats[i]
                results[i] = bench(lambda: torch.matmul(a, b), n=3, warmup=1)

        for _ in range(2):                          # 预热一轮
            ts = [threading.Thread(target=worker, args=(i, d)) for i, d in enumerate(cards)]
            [t.start() for t in ts]
            [t.join() for t in ts]
        torch.npu.synchronize()

        t0 = time.perf_counter()
        ts = [threading.Thread(target=worker, args=(i, d)) for i, d in enumerate(cards)]
        [t.start() for t in ts]
        [t.join() for t in ts]
        wall = time.perf_counter() - t0
        wall /= 3.0                                  # worker 内部跑 3 次
        agg = len(cards) * 2 * n ** 3 / wall / 1e12
        single = 2 * n ** 3 / max(results) / 1e12
        print(f"  {name.upper()}：双卡墙钟 {wall * 1e3:8.2f} ms  → 聚合 {agg:7.1f} TFLOPS"
              f"（单卡 {single:.1f}，扩展效率 {agg / (2 * single) * 100:.0f}%）")
        for a, b in mats:
            del a, b
        torch.npu.empty_cache()
    print()

    # ---------- 汇总 ----------
    print("=== 汇总（实测）===")
    for name, _ in dtypes:
        tf = best[name]
        print(f"  {name.upper()}：单卡峰值 {tf:7.1f} TFLOPS（{tf / 1e3:.3f} PFLOPS）"
              f" | 整机 2 卡约 {2 * tf:7.1f} TFLOPS（{2 * tf / 1e3:.3f} PFLOPS）")


if __name__ == "__main__":
    with torch.no_grad():
        main()
