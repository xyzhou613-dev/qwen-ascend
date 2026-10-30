"""★ 决定最大一笔候选：把 scale 乘从 16.4M 元素降到 N×G

现状（生产）：`wlo = lo * s`（16.4M 次逐元素乘，msProf 里就是那 25.8 ms 的 Mul），
            再 `bmm(xlo, wlo^T)`。
候选（分组）：`A[k,n,g] = Σ_{j∈g} lo[k,n,g,j]·x[k,g,j]`，再 `y = Σ_g s[k,n,g]·A[k,n,g]`。
            —— 完全不物化反量化权重，scale 乘法从 16.4M 降到 k·N·G = 256K。

代价：分组点积要么用 einsum（batch=G=20），要么用 batch=200 的 bmm。**效率未知，先量。**

用法：python tests/t_grouped_deq.py [次数]
"""
import sys
import time

import torch
import torch_npu  # noqa: F401

KE, N, K = 10, 1280, 2560
S, J = K // 128, 64          # 20 组 × 64
DEV = "npu:0"
REP = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def tm(fn, n=REP, warm=1):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main():
    torch.manual_seed(0)
    lo = torch.randint(0, 16, (KE, N, K // 2), dtype=torch.uint8, device=DEV).to(torch.bfloat16)
    lo4 = lo.view(KE, N, S, J)                       # 对应 lo 半（每块 64 个）
    s1 = (torch.rand(KE, N, S, 1, device=DEV) * 0.05 + 0.01).to(torch.bfloat16)
    x = torch.randn(KE, K // 2, device=DEV).to(torch.bfloat16)
    x4 = x.view(KE, S, J)

    def prod():                                      # 生产：物化 w 再 bmm
        w = (lo4 * s1).reshape(KE, N, S * J)
        return torch.bmm(x.unsqueeze(1), w.transpose(1, 2)).squeeze(1)

    def grouped_einsum():                            # 候选 A：einsum（batch=G）
        A = torch.einsum("kngj,kgj->kng", lo4, x4)   # [k,N,G]
        return (A * s1.squeeze(-1)).sum(-1)

    def grouped_bmm():                               # 候选 B：batch=k*G 的 bmm
        lb = lo4.permute(0, 2, 1, 3).reshape(KE * S, N, J)      # [k*G, N, j]
        xb = x4.reshape(KE * S, J).unsqueeze(-1)                 # [k*G, j, 1]
        A = torch.bmm(lb, xb).squeeze(-1).view(KE, S, N).permute(0, 2, 1)   # [k,N,G]
        return (A * s1.squeeze(-1)).sum(-1)

    print(f"形状 k={KE} N={N} K={K}（G={S}×{J}），每候选 {REP} 次\n", flush=True)
    ref = None
    for name, fn in (("生产：物化 w + bmm", prod),
                     ("候选A：einsum('kngj,kgj->kng')", grouped_einsum),
                     ("候选B：batch=k*G 的 bmm", grouped_bmm)):
        try:
            y = fn()
            t = tm(fn)
        except Exception as ex:                      # noqa: BLE001
            print(f"  {name:34s} 失败: {type(ex).__name__}: {str(ex)[:70]}", flush=True)
            continue
        if ref is None:
            ref = y
            print(f"  {name:34s} {t:8.3f} ms   （基准）", flush=True)
        else:
            d = float((y - ref).abs().mean() / (ref.abs().mean() + 1e-12))
            print(f"  {name:34s} {t:8.3f} ms   {(t-tm(prod))/tm(prod)*100:+6.1f}% vs 生产"
                  f"   数值差异 rel {d*100:.3f}%", flush=True)
    print("\n判据：快 ≥3% 且相对误差 <0.5% 才值得继续；否则这条也放弃。", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
