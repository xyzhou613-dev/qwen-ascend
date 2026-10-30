"""★ prefill 瓶颈评估 A：把一层的成本拆开（attention / MoE / mixer / 反量化地板）。

为什么必须量：chunked prefill（一次算 B 个 token）能不能快，取决于当前每 token 的 143 ms 里
**多少是"每个 token 都要重做一遍"的固定成本**（权重反量化/搬运/发射），多少是真正的计算。
只有固定成本占比高，批量才有大收益。

做法（不用改生产代码，全部 monkeypatch）：
  A 基线：eager（不捕获）跑完整 48 层，量每 token
  B 关 attention：把 `_deltanet`/`_qsa` 换成恒等
  C 关 MoE：把 `_moe` 换成全零
  D 反量化地板：逐层对稠密权重做一次 `E.lin`（x 只有 1 个 token），量"只为反量化付的钱"
  E 专家地板：对固定 top-10 专家做两次 `ExpQ.matmul`

⚠️ 必须 `FNX_NLEARLY_NO_CAPTURE`——捕获过的图里 op 已经固定，monkeypatch 不会生效。
"""
import os
import time

import torch

from fnx import engine as E
from fnx.model import FnModel

LPC = int(os.environ.get("PR_LPC", "46"))
NL = int(os.environ.get("PR_NL", "48"))
TOK = 18643  # 任意一个普通 token


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
    os.environ["FNX_NO_CAPTURE"] = "1"
    m = FnModel(P=2048, lpc=LPC, dev0="npu:0", dev1="npu:1", nlayers=NL,
                with_head=True, with_ngram=False, verbose=False)
    print(f"  装配完成 dev0={len(m.layers0)} dev1={len(m.layers1)} 捕获={m.g0 is not None}",
          flush=True)
    m.reset()

    def step():
        m.step(TOK, with_head=True)

    t_all = tm(step, n=5, warm=2)
    print(f"\n=== A 基线（eager 全模型）: {t_all:.1f} ms/token ===", flush=True)

    # ---- B: 关 attention（linear 层用 _deltanet，full 层用 _qsa）----
    saved = []
    for L in m.layers0 + m.layers1:
        if L.ltype in ("linear", "ple"):
            saved.append((L, "_deltanet", L._deltanet))
            L._deltanet = lambda h: h
        else:
            saved.append((L, "_qsa", L._qsa))
            L._qsa = lambda h, pos: h
    t_noattn = tm(step, n=5, warm=1)
    for L, nm, fn in saved:
        setattr(L, nm, fn)
    print(f"  B 关 attention          : {t_noattn:6.1f} ms  "
          f"(attention ≈ {t_all - t_noattn:5.1f} ms, {(t_all-t_noattn)/t_all*100:4.1f}%)",
          flush=True)

    # ---- C: 关 MoE ----
    saved = []
    for L in m.layers0 + m.layers1:
        saved.append((L, L._moe))
        L._moe = lambda x: torch.zeros_like(x)
    t_nomoe = tm(step, n=5, warm=1)
    for L, fn in saved:
        L._moe = fn
    print(f"  C 关 MoE                : {t_nomoe:6.1f} ms  "
          f"(MoE ≈ {t_all - t_nomoe:5.1f} ms, {(t_all-t_nomoe)/t_all*100:4.1f}%)",
          flush=True)

    # ---- D: 反量化地板（自动发现该层所有 int8 权重，x 只 1 token；输入维按 W.K 取）----
    L0 = m.layers0[0]
    names = [a for a in dir(L0) if not a.startswith("_")]
    ws = []
    for a in names:
        try:
            W = getattr(L0, a)
        except Exception:
            continue
        if hasattr(W, "q") and hasattr(W, "K"):
            ws.append((a, W))
    print(f"      发现 {len(ws)} 个 int8 稠密权重：", flush=True)
    tot_dq = 0.0
    rows = []
    for a, W in ws:
        if getattr(W, "q", None) is None:
            print(f"      {a:12s} 跳过（q=None，非 int8 路径）", flush=True)
            continue
        xx = torch.zeros(1, W.K, dtype=torch.bfloat16, device=W.q.device)
        tt = tm(lambda W=W, xx=xx: E.lin(xx, W), n=5, warm=1)
        tot_dq += tt
        rows.append((tt, a, W))
    for tt, a, W in sorted(rows, reverse=True):
        print(f"      {a:12s} {tt:6.2f} ms  (int8 {W.q.numel()/2**20:6.1f} MiB, "
              f"K={W.K} N={W.q.shape[0]})", flush=True)
    # lm_head 反量化（每 token 一次）
    hd = m.lm_head.q.device
    xf = torch.zeros(1, E.H, dtype=torch.bfloat16, device=hd)
    t_head = tm(lambda: E.lin(xf, m.lm_head), n=3, warm=1)
    print(f"      lm_head      {t_head:6.2f} ms  (int8 {m.lm_head.q.numel()/2**20:6.1f} MiB,"
          f" 分块 {m.lm_head.chunk_n})", flush=True)
    # 跨卡中转
    t_relay = tm(lambda: (m.h_host.copy_(m.h1), m.h1.copy_(m.h_host)), n=5, warm=1)
    print(f"      跨卡中转 h1->host->h1 {t_relay:6.2f} ms", flush=True)

    print(f"   D 单层稠密 int8 权重反量化地板 ≈ {tot_dq:6.2f} ms × 48 层 = {tot_dq*48:6.1f} ms/token"
          f"（占基线 {tot_dq*48/t_all*100:.1f}%）", flush=True)
    print(f"     + lm_head {t_head:.2f} ms（每 token 一次，不乘层数）", flush=True)

    # ---- E: 专家地板（固定 top-10）----
    ids = torch.arange(E.TOPK, dtype=torch.int64, device="npu:0")
    x2 = torch.zeros(E.TOPK, 1, E.H, dtype=torch.bfloat16, device="npu:0")
    t_gu = tm(lambda: L0.exp_gu.matmul(x2, ids), n=5, warm=1)
    act = torch.zeros(E.TOPK, 1, L0.exp_dn.K, dtype=torch.bfloat16, device="npu:0")
    t_dn = tm(lambda: L0.exp_dn.matmul(act, ids), n=5, warm=1)
    print(f"   E 单层专家（top-10）   ≈ {t_gu + t_dn:6.2f} ms "
          f"(gu {t_gu:.2f} [K={L0.exp_gu.K} N={L0.exp_gu.N}] + dn {t_dn:.2f} "
          f"[K={L0.exp_dn.K} N={L0.exp_dn.N}]) × 48 = {(t_gu+t_dn)*48:6.1f} ms/token",
          flush=True)

    print("\n=== 生产口径换算（捕获后 142.9 ms/token，eager 基线的 "
          f"{142.9/t_all*100:.0f}%）===", flush=True)
    sc = 142.9 / t_all
    print(f"  attention ≈ {(t_all-t_noattn)*sc:5.1f} ms   MoE ≈ {(t_all-t_nomoe)*sc:5.1f} ms"
          f"   mixer/其它 ≈ {(t_all-(t_all-t_noattn)-(t_all-t_nomoe))*sc:5.1f} ms", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
