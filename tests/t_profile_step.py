"""★ 剖面：把一步 decode 拆成 5 段分别计时，找出 343 ms 花在哪。

用小模型（NL=8, LPC=4）跑，单层 eager 成本与层数无关，可以线性外推到 46 层。
每段都用 device 同步包住，避免异步掩盖。

阶段：
  1 embed   ：embed_row（宿主反量化 + H2D）
  2 g0      ：dev0 半边的图 replay
  3 relay   ：h0 -> h_host -> h1（跨卡中转）
  4 body1   ：dev1 半边 eager（按层计时）
  5 head    ：mixer + lm_head（16 个 chunk 的反量化+matmul）
"""
import os
import time

import torch

from fnx import engine as E
from fnx.model import FnModel

NL = int(os.environ.get("PF_NL", "8"))
LPC = int(os.environ.get("PF_LPC", "4"))
STEPS = int(os.environ.get("PF_STEPS", "8"))


def t(dev="npu:0"):
    torch.npu.synchronize(dev)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/models/Qwen3.8-Flash-Next")
    ids = tok("昇腾处理器面向大规模深度学习训练与推理场景", return_tensors="pt").input_ids[0, :8].tolist()

    m = FnModel(P=2048, lpc=LPC, dev0="npu:0", dev1="npu:1", nlayers=NL,
                with_head=True, with_ngram=False, verbose=False)
    print(f"=== 剖面 NL={NL} LPC={LPC}  dev0={len(m.layers0)} 层  dev1={len(m.layers1)} 层 ===",
          flush=True)
    m.capture()
    print(f"    图：g0={'有' if m.g0 is not None else '无'} g1={'有' if m.g1 is not None else '无'}",
          flush=True)
    m.reset()

    # 预热
    for i in range(3):
        m.step(int(ids[i]), with_head=True)
    m.reset()

    acc = {k: 0.0 for k in ("embed", "g0", "relay", "body1", "head", "total")}
    layer_acc = {i: 0.0 for i in range(len(m.layers1))}

    for s in range(STEPS):
        tokid = int(ids[s % len(ids)])
        t0 = time.perf_counter()
        t()
        # 1 embed
        a = time.perf_counter()
        dr = m.h1.device
        m.tok.fill_(tokid)
        m.emb_buf.view(-1).copy_(m.embed_row(tokid).view(-1))
        t()
        b = time.perf_counter(); acc["embed"] += b - a
        # 2 g0
        m.g0.replay()
        t()
        c = time.perf_counter(); acc["g0"] += c - b
        # 3 relay
        m.h_host.copy_(m.h0)
        m.h1.copy_(m.h_host)
        t(dr)
        d = time.perf_counter(); acc["relay"] += d - c
        # 4 body1 逐层
        e = d
        h = m.h1
        for i, L in enumerate(m.layers1):
            h = L.forward(h, m.pos1)
            t(dr)
            f = time.perf_counter()
            layer_acc[i] += f - e
            e = f
        m.h1.copy_(h)
        t(dr)
        g = time.perf_counter(); acc["body1"] += g - d
        # 5 head
        m.head()
        t("npu:0")
        h2 = time.perf_counter(); acc["head"] += h2 - g
        m.pos0.add_(1)
        m.pos1.add_(1)
        acc["total"] += h2 - t0

    print(f"\n  {'阶段':8s} {'总计(ms)':>10s} {'每步(ms)':>10s}")
    for k in ("embed", "g0", "relay", "body1", "head", "total"):
        print(f"  {k:8s} {acc[k]*1e3:10.2f} {acc[k]/STEPS*1e3:10.2f}")
    print(f"\n  body1 逐层（ms/步）：")
    for i, v in layer_acc.items():
        print(f"    L{i}: {v/STEPS*1e3:7.3f}")
    per = sum(layer_acc.values()) / STEPS * 1e3 / max(1, len(m.layers1))
    print(f"\n  ★ dev1 每层平均 {per:.3f} ms/步 —— 外推到 22 层 = {per*22:.1f} ms")
    fixed = (acc["total"] - acc["body1"]) / STEPS * 1e3
    print(f"  ★ 固定开销（embed+g0+relay+head）{fixed:.1f} ms/步")
    print(f"  ★ 外推 46 层（24/22）：{per*22 + fixed:.1f} ms/token")
    print(f"  ★ 对照：实测 46 层 = 343.25 ms/token；旧全捕获 = 57.2 ms")


if __name__ == "__main__":
    with torch.no_grad():
        main()
