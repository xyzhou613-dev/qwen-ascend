"""t_split_r37: r37 配置下的成本拆分（第 37 轮后重测）。

与 t_prefill_split.py 的区别：
  * 专家段用 ExpW4（aclnn per-group W4A16），不再走 ExpQ.matmul；
  * 稠密反量化地板测**没有预反量化**的层（默认第 12 层，FNX_DEQ_ATTN=10 之后），
    否则量到的是 bf16 常驻权重，全是 0。

用法（容器内，服务停掉）：
  FNX_MOE_W4A16=1 FNX_PACKED_TORCH=1 FNX_EXP_G=64 FNX_LQP=... FNX_DEQ_*=1 \
  python tests/t_split_r37.py
"""
import os
import time

import torch

from fnx import engine as E
from fnx.model import FnModel

LPC = int(os.environ.get("PR_LPC", "45"))
NL = int(os.environ.get("PR_NL", "48"))
DL = int(os.environ.get("PR_DL", "12"))     # 稠密地板用哪一层（>FNX_DEQ_ATTN）
TOK = 18643


def tm(fn, n=5, warm=1):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main():
    os.environ["FNX_NO_CAPTURE"] = "1"
    m = FnModel(P=2048, lpc=LPC, dev0="npu:0", dev1="npu:1", nlayers=NL,
                with_head=True, with_ngram=False, verbose=False)
    print(f"  dev0={len(m.layers0)} dev1={len(m.layers1)} 捕获={m.g0 is not None} "
          f"专家容器={type(m.layers0[0].exp_gu).__name__}", flush=True)
    m.reset()

    def step():
        m.step(TOK, with_head=True)

    t_all = tm(step, n=5, warm=2)
    print(f"\n=== A 基线（eager，{NL} 层）: {t_all:.1f} ms/token ===", flush=True)
    print(f"    生产图捕获口径另算（本机实测 TPOT 76.7 ms）", flush=True)

    saved = []
    for L in m.layers0 + m.layers1:
        if L.ltype in ("linear", "ple"):
            saved.append((L, "_deltanet", L._deltanet))
            L._deltanet = lambda h: h
        else:
            saved.append((L, "_qsa", L._qsa))
            L._qsa = lambda h, pos: h
    t_noattn = tm(step, n=5, warm=1)
    for L, nm_, fn in saved:
        setattr(L, nm_, fn)
    print(f"  B 关 attention : {t_noattn:6.1f} ms  (attention ≈ {t_all-t_noattn:5.1f} ms, "
          f"{(t_all-t_noattn)/t_all*100:4.1f}%)", flush=True)

    saved = []
    for L in m.layers0 + m.layers1:
        saved.append((L, L._moe))
        L._moe = lambda x: torch.zeros_like(x)
    t_nomoe = tm(step, n=5, warm=1)
    for L, fn in saved:
        L._moe = fn
    print(f"  C 关 MoE       : {t_nomoe:6.1f} ms  (MoE ≈ {t_all-t_nomoe:5.1f} ms, "
          f"{(t_all-t_nomoe)/t_all*100:4.1f}%)", flush=True)

    # D) 稠密反量化地板：只统计还没预反量化的 int8 权重
    L0 = m.layers0[DL]
    rows = []
    for a in dir(L0):
        if a.startswith("_"):
            continue
        try:
            W = getattr(L0, a)
        except Exception:  # noqa: BLE001
            continue
        if hasattr(W, "q") and hasattr(W, "K") and getattr(W, "q", None) is not None:
            xx = torch.zeros(1, W.K, dtype=torch.bfloat16, device=W.q.device)
            rows.append((tm(lambda W=W, xx=xx: E.lin(xx, W), n=5, warm=1), a, W))
    tot_dq = 0.0
    print(f"  D 第 {DL} 层仍未预反量化的 int8 权重：", flush=True)
    for tt, a, W in sorted(rows, reverse=True):
        tot_dq += tt
        print(f"      {a:12s} {tt:6.3f} ms  (int8 {W.q.numel()/2**20:6.1f} MiB, "
              f"K={W.K} N={W.q.shape[0]})", flush=True)
    if not rows:
        print("      （该层全部预反量化）", flush=True)
    print(f"      合计 {tot_dq:.3f} ms/层 × {len(m.layers0)} 层 = {tot_dq*len(m.layers0):.1f} ms", flush=True)

    # E) 专家段（新容器）
    ids = torch.arange(E.TOPK, dtype=torch.int64, device="npu:0")
    gu, dn = L0.exp_gu, L0.exp_dn
    x1 = torch.zeros(1, E.H, dtype=torch.bfloat16, device="npu:0")
    a1 = torch.zeros(E.TOPK, E.MI, dtype=torch.bfloat16, device="npu:0")
    if hasattr(gu, "apply"):
        t_gu = tm(lambda: gu.apply(x1, ids))
        t_dn = tm(lambda: dn.apply_diag(a1, ids))
        print(f"  E 单层专家（ExpW4，拼接+对角）: gu {t_gu:.3f} + dn {t_dn:.3f} = "
              f"{t_gu+t_dn:.3f} ms × {NL} = {(t_gu+t_dn)*NL:.1f} ms", flush=True)
    else:
        x2 = torch.zeros(E.TOPK, 1, E.H, dtype=torch.bfloat16, device="npu:0")
        act = torch.zeros(E.TOPK, 1, dn.K, dtype=torch.bfloat16, device="npu:0")
        t_gu = tm(lambda: gu.matmul(x2, ids))
        t_dn = tm(lambda: dn.matmul(act, ids))
        print(f"  E 单层专家（ExpQ 旧路径）: {t_gu+t_dn:.3f} ms × {NL} = "
              f"{(t_gu+t_dn)*NL:.1f} ms", flush=True)

    print(f"\n  换算（captured 76.7 ms / eager {t_all:.1f} ms，比例 {76.7/t_all:.2f}）：", flush=True)
    sc = 76.7 / t_all
    print(f"    attention ≈ {(t_all-t_noattn)*sc:5.1f} ms   MoE ≈ {(t_all-t_nomoe)*sc:5.1f} ms"
          f"   其它 ≈ {(t_all-(t_all-t_noattn)-(t_all-t_nomoe))*sc:5.1f} ms", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
