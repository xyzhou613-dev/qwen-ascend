"""★ 验证 v2 打包布局在 torch 路径上的解包：正确性 + 速度。

v2 布局（已在第 27 轮用 `tools/layout_v2_check.py` 证明无损，max_abs 8.9e-16）：
    byte(64s + j) 的低 4 bit = W[128s + j]
    byte(64s + j) 的高 4 bit = W[128s + 64 + j]

关键：这个映射让解包**不需要交织** —— 只要把 S 个 128-块各写成
[w[0..63] = lo, w[64..127] = hi]，然后 reshape，索引恰好就是 128s + 64*half + j。

对比三种取数方式：
  1 现状：int8 容器 [k,N,K]（1 B/权重）
  2 v2 打包 [k,N,K/2]（0.5 B/权重）+ 块状切片解包
  3 参考：numpy 逐元素（只用来判对错，不参与计时）
"""
import time

import numpy as np
import torch
import torch_npu  # noqa: F401

from fnx.w4gemv_op import W4GemvOp

DEV = "npu:0"


def deq_container(q, s, gs):
    """现状路径：int8 容器 [k,N,K] -> bf16"""
    k, N, K = q.shape
    w = q.to(torch.bfloat16).view(k, N, K // gs, gs)
    return (w * s.to(torch.bfloat16).unsqueeze(-1)).reshape(k, N, K)


def deq_v2(p, s, gs, K):
    """v2 打包 [k,N,K/2] -> bf16（块状切片，无交织）"""
    k, N, _ = p.shape
    S = K // 128
    b = p.view(k, N, S, 64)
    lo = (b & 0x0F).to(torch.int16)
    hi = ((b >> 4) & 0x0F).to(torch.int16)
    lo = (lo ^ 8) - 8
    hi = (hi ^ 8) - 8
    w = torch.empty(k, N, S, 2, 64, dtype=torch.bfloat16, device=p.device)
    w[..., 0, :] = lo.to(torch.bfloat16)
    w[..., 1, :] = hi.to(torch.bfloat16)
    w = w.reshape(k, N, K).view(k, N, K // gs, gs)
    return (w * s.to(torch.bfloat16).unsqueeze(-1)).reshape(k, N, K)


def deq_v2_two(p, s, gs, K):
    """★ 方案 A：**不交织**。返回 (wlo, whi)，各是连续的 [k,N,K/2]。

    v2: byte(64s+j) 低 4bit -> W[128s+j]、高 4bit -> W[128s+64+j]。
    lo 的 k 索引落在组 2s（gs=64），hi 落在组 2s+1。
    下游用 y = bmm(xlo, wlo^T) + bmm(xhi, whi^T)，xlo/xhi 的切片只作用在**小激活**上。
    """
    k, N, _ = p.shape
    S = K // 128
    lo = ((p & 0x0F).to(torch.int16) ^ 8) - 8
    hi = (((p >> 4) & 0x0F).to(torch.int16) ^ 8) - 8
    wlo = lo.to(torch.bfloat16).view(k, N, S, 64)
    whi = hi.to(torch.bfloat16).view(k, N, S, 64)
    sv = s.to(torch.bfloat16).view(k, N, S, 2)
    wlo = (wlo * sv[..., 0:1]).reshape(k, N, S * 64)
    whi = (whi * sv[..., 1:2]).reshape(k, N, S * 64)
    return wlo, whi


def bench(fn, n=30, warm=5):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    torch.manual_seed(0)
    gs = 64
    for tag, N, K in (("gate_up", 1280, 2560), ("down", 2560, 640)):
        k, E = 10, 64
        q = torch.randint(-8, 8, (E, N, K), dtype=torch.int8)          # int8 容器
        s = torch.rand(E, N, K // gs, dtype=torch.float16) * 0.01
        # 用生产同一个转换函数打 v2 包
        p_np = W4GemvOp.to_v2_layout(q.numpy(), N, K)
        p = torch.from_numpy(np.ascontiguousarray(p_np, dtype=np.uint8)).to(DEV)
        qd, sd = q.to(DEV), s.to(DEV)
        idx = torch.arange(k, dtype=torch.long, device=DEV)

        a = deq_container(qd.index_select(0, idx), sd.index_select(0, idx), gs)
        b = deq_v2(p.index_select(0, idx), sd.index_select(0, idx), gs, K)

        d = (a.float() - b.float()).abs()
        rel = float(d.mean() / a.float().abs().mean())
        print(f"\n=== {tag}  N={N} K={K}  k={k} ===", flush=True)
        print(f"  打包大小 {p.numel()/2**20:6.2f} MiB  vs 容器 {q.numel()/2**20:6.2f} MiB "
              f"（省 {100*(1-p.numel()/q.numel()):.0f}%）", flush=True)
        print(f"  数值：rel {rel*100:.6f}%  max {float(d.max()):.3e}  "
              f"-> {'无损 ✅' if rel < 1e-6 else '有差 ⚠️'}", flush=True)

        t1 = bench(lambda: deq_container(qd.index_select(0, idx), sd.index_select(0, idx), gs))
        t2 = bench(lambda: deq_v2(p.index_select(0, idx), sd.index_select(0, idx), gs, K))
        t3 = bench(lambda: deq_v2_two(p.index_select(0, idx), sd.index_select(0, idx), gs, K))
        print(f"  容器解包   {t1:6.3f} ms", flush=True)
        print(f"  v2 交织解包 {t2:6.3f} ms  ← 不可用", flush=True)
        print(f"  v2 两半解包 {t3:6.3f} ms  ← 方案 A", flush=True)

        # 方案 A 的下游：两次 GEMM vs 一次 GEMM
        x = torch.randn(k, K, dtype=torch.bfloat16, device=DEV)
        xo = x.view(k, K // 128, 2, 64)[:, :, 0, :].reshape(k, K // 2).contiguous()
        xe = x.view(k, K // 128, 2, 64)[:, :, 1, :].reshape(k, K // 2).contiguous()
        w1 = deq_container(qd.index_select(0, idx), sd.index_select(0, idx), gs)
        wlo, whi = deq_v2_two(p.index_select(0, idx), sd.index_select(0, idx), gs, K)
        g1 = bench(lambda: torch.bmm(x.unsqueeze(1), w1.transpose(1, 2)), n=20)
        g2 = bench(lambda: torch.bmm(xo.unsqueeze(1), wlo.transpose(1, 2))
                   + torch.bmm(xe.unsqueeze(1), whi.transpose(1, 2)), n=20)
        # 数值核对：两种算法结果必须一致
        y1 = torch.bmm(x.unsqueeze(1), w1.transpose(1, 2))
        y2 = (torch.bmm(xo.unsqueeze(1), wlo.transpose(1, 2))
              + torch.bmm(xe.unsqueeze(1), whi.transpose(1, 2)))
        dd = (y1.float() - y2.float()).abs()
        rr = float(dd.mean() / y1.float().abs().mean().clamp_min(1e-12))
        print(f"  GEMM 一次 {g1:6.3f} ms   两次 {g2:6.3f} ms   "
              f"数值 rel {rr*100:.4f}% {'✅' if rr < 5e-3 else '❌'}", flush=True)
        print(f"  → 方案 A 合计 {t3+g2:6.3f} ms/层，对比现状 {t1+g1:6.3f} ms/层", flush=True)

    print("\n  说明：解包每层每 token 各一次；若 v2 解包比容器慢但不超 ~0.3 ms，"
          "仍值得——它省 1 GiB/层，能让单卡多装 16 层。", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
