"""t_prefill_chunk: prefill 分块（T 个 token 一次算）能摊薄多少 —— 真实权重测系数。

回答一个问题：**prefill 的每 token 成本里，多少按 T 摊薄、多少必须每 token 重做？**

做法（不新建层实现，只测三个块的实际成本）：
  A 稠密块：L00 的全部稠密权重（DenseW8 / bf16 Raw），X=[T,H] 一次 lin → 每 token 成本
  B MoE 权重：固定 10 个专家，X=[T,H] 走同一条 [K, 10*N/8] 拼接 + 一次 WQBM
     （这是 MoE 摊薄的**上界**：真实分块里 T 个 token 的专家集合不同）
  C MoE 并集 gather：T 个 token 各取 10 个专家 → 去重后 U 个，量 index_select 的代价
  D 残差：现存每 token 成本 − A(T=1) − B(T=1) ⇒ 注意力/递推/逐元素，这部分**不随 T 摊薄**

用法（容器内，服务停掉）：
  FNX_DENSE_W8=1 FNX_MOE_W4A16=1 ... python tests/t_prefill_chunk.py
"""
import os
import time

import torch

from fnx import engine as E
from fnx.load import load_layer, QDIR

DEV = "npu:0"
TS = [1, 8, 32]
NEXP_PER_TOK = E.TOPK


def tm(fn, n=3, warm=1):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main():
    W = load_layer(os.path.join(QDIR, "L00.safetensors"), 0, DEV)
    dense = []
    for k, w in W.items():
        if k.startswith("mlp.experts"):
            continue
        if isinstance(w, (E.DenseW8, E.QW, E.Raw)):
            dense.append((k, w))
        elif torch.is_tensor(w) and w.dim() == 2:   # 卷积/小张量（3D）不算稠密 GEMM
            dense.append((k, w))
    print(f"L00 载入：{len(W)} 项，其中稠密 {len(dense)} 项", flush=True)
    xs = {T: torch.randn(T, E.H, dtype=torch.bfloat16, device=DEV) for T in TS}

    # A) 稠密块：按输入维 K 分组（w_up 吃 320，其余吃 2560）
    def wK(w):
        if isinstance(w, (E.DenseW8, E.QW)):
            return w.K
        if isinstance(w, E.Raw):
            return w.w.shape[-1]
        return w.shape[-1]

    groups = {}
    for k, w in dense:
        groups.setdefault(wK(w), []).append((k, w))
    print("  按 K 分组：" + ", ".join(f"K={K}:{len(v)}个" for K, v in sorted(groups.items())),
          flush=True)
    print("\n=== A 稠密块（全部稠密权重各一次 lin）===", flush=True)
    base = None
    for T in TS:
        tot = 0.0
        for K, ws in sorted(groups.items()):
            x = torch.randn(T, K, dtype=torch.bfloat16, device=DEV)
            tot += tm(lambda x=x, ws=ws: [E.lin(x, w) for _, w in ws])
        per = tot / T
        if base is None:
            base = per
        print(f"  T={T:3d}  整块 {tot:8.3f} ms   每 token {per:8.3f} ms   (相对 T=1: {base/per:5.1f}x)",
              flush=True)

    # B) MoE 权重（固定专家集合，测上界摊薄）
    print("\n=== B MoE 权重（固定 10 专家，一次 WQBM）===", flush=True)
    gu = W["mlp.experts.gate_up_proj"]
    dn = W["mlp.experts.down_proj"]
    ids = torch.arange(NEXP_PER_TOK, dtype=torch.int64, device=DEV)
    wgu = gu._wcat(ids)
    sgu = gu._scat(ids)
    base = None
    for T in TS:
        x = xs[T]
        t = tm(lambda x=x: gu.apply(x, ids))
        per = t / T
        if base is None:
            base = per
        print(f"  gu T={T:3d}  整块 {t:8.3f} ms   每 token {per:8.3f} ms   ({base/per:5.1f}x)",
              flush=True)
    ids_t = torch.arange(NEXP_PER_TOK * max(TS), dtype=torch.int64, device=DEV) % dn.E
    base = None
    for T in TS:
        act = torch.randn(T * NEXP_PER_TOK, E.MI, dtype=torch.bfloat16, device=DEV)
        idx = ids_t[: T * NEXP_PER_TOK]
        t = tm(lambda act=act, idx=idx: dn.apply(act, idx))
        per = t / T
        if base is None:
            base = per
        print(f"  dn T={T:3d}  整块 {t:8.3f} ms   每 token {per:8.3f} ms   ({base/per:5.1f}x)",
              flush=True)

    # C) 并集 gather（T 个 token × 10 专家，去重后）
    print("\n=== C MoE 并集 gather（去重 U 个专家）===", flush=True)
    torch.manual_seed(0)
    tot_experts = gu.E
    for T in TS:
        pick = torch.randint(0, tot_experts, (T, NEXP_PER_TOK), device=DEV)
        flat = pick.reshape(-1)
        uq = torch.unique(flat)
        U = int(uq.numel())
        t = tm(lambda uq=uq: (gu.w4.index_select(1, uq), gu.s8t.index_select(1, uq)))
        print(f"  T={T:3d}  行数={T*NEXP_PER_TOK}  去重 U={U:3d}  gather {t:8.3f} ms  "
              f"每 token {t/T:8.3f} ms", flush=True)

    # D) 残差：现存每 token 成本 − A(T=1) − B(T=1)
    print("\n=== D 残差（注意力/递推/逐元素，不随 T 摊薄）===", flush=True)
    print("  用上面的 A/B 的 T=1 值 + 实测的生产每 token 设备时间（预填 ~44 ms，无 head）",
          flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
