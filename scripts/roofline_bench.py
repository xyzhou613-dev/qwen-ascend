#!/usr/bin/env python3
"""roofline_bench.py — 在独占卡上量 910B4 的三个天花板，供 Roofline 分析用。

1) 计算天花板：bf16 matmul 在 1024/2048/4096/8192 的实测 TFLOPS（torch_npu.matmul）
2) 带宽天花板：HBM 拷贝 / 读 的有效 GB/s
3) 本项目实际用的算子：npu_weight_quant_batchmatmul（int4 权重）在不同 M 下的 TFLOPS 与等效 GB/s
4) 逐元素（Vector）吞吐：mul / silu 链的 GB/s

用法（容器内、独占卡）：
  /root/ascend-dev/venv/bin/python roofline_bench.py
"""
import time

import torch
import torch_npu  # noqa: F401

DEV = "npu:0"


def bench(fn, n=5, warmup=2):
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n


def fmt(x):
    return f"{x:,.1f}".replace(",", " ")


def main():
    print(f"device={DEV}  " + torch.npu.get_device_name(0) if hasattr(torch.npu, "get_device_name") else f"device={DEV}")
    print()

    # ---------- 1) bf16 matmul 计算天花板 ----------
    print("=== 1) bf16 matmul（计算天花板）===")
    print(f"{'N':>6} {'ms':>9} {'TFLOPS':>9}")
    peak = 0.0
    for n in (1024, 2048, 4096, 8192):
        a = torch.randn(n, n, device=DEV, dtype=torch.bfloat16)
        b = torch.randn(n, n, device=DEV, dtype=torch.bfloat16)
        ms = bench(lambda: torch.matmul(a, b), n=3 if n >= 8192 else 5) * 1e3
        tflops = 2 * n ** 3 / (ms / 1e3) / 1e12
        peak = max(peak, tflops)
        print(f"{n:>6} {ms:>9.2f} {tflops:>9.1f}")
        del a, b
        torch.npu.empty_cache()
    print(f"  ⇒ 实测计算天花板 ≈ {peak:.1f} TFLOPS (bf16)")
    print()

    # ---------- 2) HBM 带宽天花板 ----------
    print("=== 2) HBM 带宽（拷贝 / 读）===")
    for mb in (256, 1024, 2048):
        n = mb * 1024 * 1024 // 2          # bf16 元素数
        x = torch.ones(n, device=DEV, dtype=torch.bfloat16)
        y = torch.empty_like(x)
        ms = bench(lambda: y.copy_(x), n=3) * 1e3
        gb = 2 * n * 2 / 1e9               # 读 + 写
        print(f"  copy {mb:>5} MB: {ms:>8.2f} ms  → {gb / (ms / 1e3):>8.1f} GB/s")
        del x, y
        torch.npu.empty_cache()
    n = 1024 * 1024 * 256
    x = torch.ones(n, device=DEV, dtype=torch.bfloat16)
    ms = bench(lambda: x.sum(), n=3) * 1e3
    print(f"  read {n * 2 / 1e6:>5.0f} MB: {ms:>8.2f} ms  → {n * 2 / 1e9 / (ms / 1e3):>8.1f} GB/s（只读）")
    del x
    torch.npu.empty_cache()
    print()

    # ---------- 3) int4 权重算子（本项目主力）----------
    print("=== 3) npu_weight_quant_batchmatmul（int4 权重，真实形状 K=2560 N=1280）===")
    K, N, GS = 2560, 1280, 64
    w = torch.randint(-2 ** 30, 2 ** 30, (K, N // 8), device=DEV, dtype=torch.int32)
    scale = torch.ones(K // GS, N, device=DEV, dtype=torch.bfloat16)
    wbytes = K * N / 2
    print(f"{'M':>6} {'ms':>9} {'TFLOPS':>9} {'等效 GB/s':>11} {'权重 {:.1f} MB'.format(wbytes / 1e6):>14}")
    for M in (1, 8, 64, 512, 4096, 8192):
        x = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
        ms = bench(lambda: torch_npu.npu_weight_quant_batchmatmul(
            x, w, scale, antiquant_group_size=GS), n=3) * 1e3
        tflops = 2 * M * K * N / (ms / 1e3) / 1e12
        gbs = (wbytes + M * K * 2) / (ms / 1e3) / 1e9
        print(f"{M:>6} {ms:>9.3f} {tflops:>9.2f} {gbs:>11.1f}")
        del x
        torch.npu.empty_cache()
    print()

    # ---------- 4) Vector 逐元素吞吐 ----------
    print("=== 4) Vector 逐元素（mul / silu 链）===")
    n = 16 * 1024 * 1024                        # 16M 元素 = 32 MB
    a = torch.randn(n, device=DEV, dtype=torch.bfloat16)
    b = torch.randn(n, device=DEV, dtype=torch.bfloat16)
    ms = bench(lambda: torch.mul(a, b), n=5) * 1e3
    print(f"  mul  {n / 1e6:.0f}M 元素: {ms:>7.2f} ms → {3 * n * 2 / 1e9 / (ms / 1e3):>7.1f} GB/s")
    ms = bench(lambda: torch.nn.functional.silu(a), n=5) * 1e3
    print(f"  silu {n / 1e6:.0f}M 元素: {ms:>7.2f} ms → {2 * n * 2 / 1e9 / (ms / 1e3):>7.1f} GB/s")


if __name__ == "__main__":
    with torch.no_grad():
        main()
