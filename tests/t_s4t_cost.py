"""量 s4T 转置的代价：一次性（装配期，全模型）vs 每 token（decode 现转）。

结论用来决定 load.py 要不要把 s4 直接按 [K/gs][N] 落盘/驻留。
单层 s4 真实形状：gate_up [512, 1280, 80] + down [512, 2560, 20]，fp16，合计 0.1465 GiB（QP manifest 实测）。
"""
import time

import torch
import torch_npu  # noqa: F401

GS = 32
LAYERS = 48
# 单层两个张量（专家数, N, K/gs）
SHAPES = [("gate_up", 512, 1280, 80), ("down", 512, 2560, 20)]


def bench(fn, warm=2, rep=5):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / rep * 1e3     # ms


def main():
    print(f"[s4t] LAYERS={LAYERS} 每层 s4 = gate_up[512,1280,80] + down[512,2560,20] fp16", flush=True)
    nbytes = sum(e * n * g * 2 for _, e, n, g in SHAPES)
    print(f"[s4t] 每层 {nbytes} B = {nbytes/2**30:.4f} GiB；全模型 {LAYERS*nbytes/2**30:.2f} GiB", flush=True)

    # (a) 装配期一次性：全模型所有层的 s4 都转置
    total_once = 0.0
    for name, e, n, g in SHAPES:
        src = torch.randn(e, n, g, dtype=torch.float16, device="npu:0")
        ms_per_layer = bench(lambda: src.transpose(1, 2).contiguous())
        total_once += ms_per_layer * LAYERS
        print(f"[s4t] 装配期 {name}: {ms_per_layer:.2f} ms/层 → ×{LAYERS} 层 = {ms_per_layer*LAYERS:.0f} ms", flush=True)
        del src
    print(f"[s4t] === 装配期一次性总代价 = {total_once/1e3:.2f} s（只做一次）===", flush=True)

    # (b) 每 token 现转：只转当 token 激活的 10 个专家
    total_tok = 0.0
    for name, e, n, g in SHAPES:
        src = torch.randn(10, n, g, dtype=torch.float16, device="npu:0")
        ms = bench(lambda: src.transpose(1, 2).contiguous())
        total_tok += ms * LAYERS
        print(f"[s4t] 每token {name}: {ms:.3f} ms → ×{LAYERS} 层 = {ms*LAYERS:.2f} ms", flush=True)
        del src
    print(f"[s4t] === 每 token 现转总代价 = {total_tok:.2f} ms/token ===", flush=True)

    # 对比：kernel 本身的专家 GEMV 段
    print(f"[s4t] 对比 kernel 专家段实测 ≈ 21.8 ms/token", flush=True)
    print(f"[s4t] 现转占比 = {total_tok/21.8*100:.1f}%  → "
          f"{'必须先转置（装配期做一次）' if total_tok/21.8 > 0.05 else '现转也可接受'}", flush=True)


if __name__ == "__main__":
    main()
