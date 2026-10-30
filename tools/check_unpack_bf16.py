"""★ 方案 B：在 **bf16 算术**里解包（不用整数位运算 —— 那些在 NPU 上慢 20-500 倍）。

映射：byte(64s+j) 的 0..15 nibble n，值 = (n ^ 8) - 8 == ((n + 8) % 16) - 8
      lo -> W[128s+j]、hi -> W[128s+64+j]
"""
import time

import numpy as np
import torch
import torch_npu  # noqa: F401

from fnx.w4gemv_op import W4GemvOp

DEV = "npu:0"


def bench(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def deq_container(q, s, gs):
    k, N, K = q.shape
    w = q.to(torch.bfloat16).view(k, N, K // gs, gs)
    return (w * s.to(torch.bfloat16).unsqueeze(-1)).reshape(k, N, K)


def deq_bf16_two(p, s, gs, K):
    """返回 (wlo, whi)，两个连续 [k,N,K/2]；符号与 scale 都还原好。"""
    k, N, _ = p.shape
    S = K // 128
    pb = p.to(torch.bfloat16)
    hi = torch.floor(pb * 0.0625)                  # n_hi = floor(byte/16)
    lo = pb - hi * 16.0                            # n_lo
    lo = torch.remainder(lo + 8.0, 16.0) - 8.0     # (n^8)-8
    hi = torch.remainder(hi + 8.0, 16.0) - 8.0
    wlo = lo.view(k, N, S, 64)
    whi = hi.view(k, N, S, 64)
    sv = s.to(torch.bfloat16).view(k, N, S, 2)
    wlo = (wlo * sv[..., 0:1]).reshape(k, N, S * 64)
    whi = (whi * sv[..., 1:2]).reshape(k, N, S * 64)
    return wlo, whi


def main():
    torch.manual_seed(0)
    gs = 64
    for tag, N, K in (("gate_up", 1280, 2560), ("down", 2560, 640)):
        k, E = 10, 32
        q = torch.randint(-8, 8, (E, N, K), dtype=torch.int8)
        s = torch.rand(E, N, K // gs, dtype=torch.float16) * 0.01
        p = torch.from_numpy(np.ascontiguousarray(
            W4GemvOp.to_v2_layout(q.numpy(), N, K), dtype=np.uint8)).to(DEV)
        qd, sd = q.to(DEV), s.to(DEV)
        idx = torch.arange(k, dtype=torch.long, device=DEV)

        x = torch.randn(k, K, dtype=torch.bfloat16, device=DEV)
        xv = x.view(k, K // 128, 2, 64)
        xlo = xv[:, :, 0, :].reshape(k, K // 2).contiguous()
        xhi = xv[:, :, 1, :].reshape(k, K // 2).contiguous()

        w1 = deq_container(qd.index_select(0, idx), sd.index_select(0, idx), gs)
        wlo, whi = deq_bf16_two(p.index_select(0, idx), sd.index_select(0, idx), gs, K)
        y1 = torch.bmm(x.unsqueeze(1), w1.transpose(1, 2))
        y2 = (torch.bmm(xlo.unsqueeze(1), wlo.transpose(1, 2))
              + torch.bmm(xhi.unsqueeze(1), whi.transpose(1, 2)))
        d = (y1.float() - y2.float()).abs()
        rel = float(d.mean() / y1.float().abs().mean().clamp_min(1e-12))
        print(f"\n=== {tag} N={N} K={K} ===", flush=True)
        print(f"  数值 rel {rel*100:.4f}%  max {float(d.max()):.3e}  "
              f"-> {'一致 ✅' if rel < 5e-3 else '不一致 ❌'}", flush=True)

        t1 = bench(lambda: deq_container(qd.index_select(0, idx), sd.index_select(0, idx), gs))
        t1g = bench(lambda: torch.bmm(x.unsqueeze(1), w1.transpose(1, 2)))
        t2 = bench(lambda: deq_bf16_two(p.index_select(0, idx), sd.index_select(0, idx), gs, K))
        t2g = bench(lambda: (torch.bmm(xlo.unsqueeze(1), wlo.transpose(1, 2))
                             + torch.bmm(xhi.unsqueeze(1), whi.transpose(1, 2))))
        print(f"  容器路径：解包 {t1:6.3f} + gemm {t1g:6.3f} = {t1+t1g:6.3f} ms/层", flush=True)
        print(f"  bf16 两半：解包 {t2:6.3f} + gemm {t2g:6.3f} = {t2+t2g:6.3f} ms/层", flush=True)
        print(f"  → 差值 {(t2+t2g)-(t1+t1g):+.3f} ms/层，显存省 "
              f"{(q.numel()-p.numel())/2**30:.2f} GiB/层", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
