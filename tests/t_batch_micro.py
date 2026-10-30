"""★ 批量收益微基准：int8 稠密权重（QW）在 B=1..32 下的单次调用耗时。

逻辑：每 token 都要把整块 int8 权重反量化一次再 GEMM。若 B=8 的调用耗时远小于 8×B=1，
说明"权重读取/反量化/发射"这部分能被批量摊薄 → 多请求 batch 值得做。
若耗时随 B 线性增长，说明 B=1 时已经打满了带宽 → 批量只能提升吞吐利用率，不划算。

只分配几十 MiB 显存，**不需要停服务**。
"""
import time

import torch

from fnx import engine as E

DEV = "npu:0"
BS = (1, 2, 4, 8, 16, 32)


def mk_qw(n, k, gs=64):
    q = torch.randint(-127, 127, (n, k), dtype=torch.int8, device=DEV)
    s = ((torch.rand(n, k // gs, device=DEV) * 0.05) + 0.01).to(torch.float16)
    return E.QW(q, s, K=k)


def tm(fn, n=10, warm=3):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def run(name, n_out, k):
    W = mk_qw(n_out, k)
    print(f"\n=== {name}  (int8 {W.q.numel()/2**20:.1f} MiB, K={k} N={n_out}) ===", flush=True)
    base = None
    for B in BS:
        x = torch.zeros(B, k, dtype=torch.bfloat16, device=DEV)
        tt = tm(lambda x=x: E.lin(x, W))
        per = tt / B
        if base is None:
            base = per
        print(f"  B={B:2d}: {tt:7.2f} ms   每 token {per:6.2f} ms   "
              f"相对 B=1 加速 {base/per:4.2f}×", flush=True)
    del W
    torch.npu.empty_cache()


def main():
    # 注意力投影（最重的稠密权重）
    run("w_qkv 类", 10240, 2560)
    # 小权重（每层很多个）
    run("gate_w 类", 512, 2560)
    # lm_head 规模（每 token 一次，分块 16384）
    print("\n=== 参照：同一块权重的纯 bf16 matmul（不含反量化）===")
    n_out, k = 10240, 2560
    wb = torch.zeros(n_out, k, dtype=torch.bfloat16, device=DEV)
    for B in (1, 8):
        x = torch.zeros(B, k, dtype=torch.bfloat16, device=DEV)
        tt = tm(lambda x=x: torch.nn.functional.linear(x, wb))
        print(f"  bf16 B={B:2d}: {tt:7.2f} ms  每 token {tt/B:6.2f} ms", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
