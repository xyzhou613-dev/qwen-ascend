"""★ 再往里拆：一个 eager 层的 6 个子块分别多贵。

layer.forward 的结构：
  1 gated_residual(attn)   rms_norm + 两个 lin + sigmoid/mean
  2 attention              _deltanet 或 _qsa
  3 inject(attn)
  4 gated_residual(mlp)
  5 _moe                   router + gather_deq + 2×bmm + weighted_sum
  6 inject(mlp)

对一个 dev1 的层逐子块计时（每块前后 device 同步）。
"""
import os
import time

import torch

from fnx import engine as E
from fnx.model import FnModel

NL = int(os.environ.get("PF_NL", "8"))
LPC = int(os.environ.get("PF_LPC", "4"))
STEPS = int(os.environ.get("PF_STEPS", "6"))


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/models/Qwen3.8-Flash-Next")
    ids = tok("昇腾处理器面向大规模深度学习训练与推理场景", return_tensors="pt").input_ids[0, :8].tolist()

    m = FnModel(P=2048, lpc=LPC, dev0="npu:0", dev1="npu:1", nlayers=NL,
                with_head=True, with_ngram=False, verbose=False)
    L = m.layers1[0]
    dev = L.dev
    print(f"=== 单层子块剖面：L{L.idx} type={L.ltype} dev={dev}  "
          f"gated_residual×2 / attn / moe / inject×2 ===", flush=True)

    def sync():
        torch.npu.synchronize(dev)

    acc = {}

    def add(k, t0):
        sync()
        acc[k] = acc.get(k, 0.0) + time.perf_counter() - t0

    with torch.no_grad():
        h = m.h1.clone()
        pos = m.pos1
        # 预热
        for _ in range(3):
            hh = h
            mix, inj = E.gated_residual(hh, L.a_hc_norm, L.a_down, L.a_up, L.a_inj)
            y = L._deltanet(mix) if L.ltype in ("linear", "ple") else L._qsa(mix, pos)
            hh = E.inject(hh, inj, y)
            mix, inj = E.gated_residual(hh, L.m_hc_norm, L.m_down, L.m_up, L.m_inj)
            y = L._moe(mix)
            hh = E.inject(hh, inj, y)
        sync()

        for _ in range(STEPS):
            t0 = time.perf_counter(); mix, inj = E.gated_residual(h, L.a_hc_norm, L.a_down, L.a_up, L.a_inj); add("1 gated_res(attn)", t0)
            t0 = time.perf_counter(); y = L._deltanet(mix) if L.ltype in ("linear", "ple") else L._qsa(mix, pos); add("2 attn", t0)
            t0 = time.perf_counter(); h2 = E.inject(h, inj, y); add("3 inject(attn)", t0)
            t0 = time.perf_counter(); mix2, inj2 = E.gated_residual(h2, L.m_hc_norm, L.m_down, L.m_up, L.m_inj); add("4 gated_res(mlp)", t0)
            t0 = time.perf_counter(); y2 = L._moe(mix2); add("5 moe", t0)
            t0 = time.perf_counter(); h3 = E.inject(h2, inj2, y2); add("6 inject(mlp)", t0)
            t0 = time.perf_counter(); h = h3 + 0; add("7 其余(赋值)", t0)

    print(f"\n  {'子块':20s} {'ms/步':>9s}")
    tot = 0.0
    for k in sorted(acc):
        v = acc[k] / STEPS * 1e3
        tot += v
        print(f"  {k:20s} {v:9.3f}")
    print(f"  {'合计':20s} {tot:9.3f}")
    print(f"\n  ★ 对照：层总时间（上一步实测）≈ 12.19 ms/步")
    print(f"  ★ 外推 22 层 = {tot*22:.1f} ms/token")


if __name__ == "__main__":
    with torch.no_grad():
        main()
