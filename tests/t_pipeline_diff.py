"""★★★ 定位 FnModel 管线 bug：同一批层，三种驱动方式对比 logits。

已知事实：
  - 用同一套 `E.FnLayer` + 同一批 QP 权重，**层主序**（每层独立状态、脚本直调）48 层 = ~5.6
  - transformers 参考实现 48 层 = 5.6706
  - 但 `FnModel`（token 主序 + body0/body1 + 图捕获 + 跨卡中转）48 层 = 12.2044
→ bug 在 FnModel 的**驱动方式**里，不在层数学里。

本脚本在**浅层**就把三种驱动方式摆在一起比 logits（浅层 NLL 是平的，看不出差别，
但 logits 相对差在浅层就已经能分辨真伪）：

  A 层主序     ：每层喂完 T 个 token 再进下一层（每层自己的状态，pos=0..T-1）
  B token 主序 eager：m.step() 走 body0/body1（g0=None）
  C token 主序 图捕获：先 m.capture()，再 m.step() 走 replay

判定：A≈B 而 C≠B → 图捕获是 bug；A≠B → step()/跨卡中转是 bug。

用法（容器内，先停服）：
  cd /root/ascend-dev/fnx && PD_NL=8 PD_TOK=4 python -u tests/t_pipeline_diff.py
"""
import json
import os

import torch

from fnx import engine as E
from fnx.load import QDIR, SRCDIR
from fnx.model import FnModel

NL = int(os.environ.get("PD_NL", "8"))
NT = int(os.environ.get("PD_TOK", "4"))
LPC = int(os.environ.get("PD_LPC", str(NL)))      # 放在卡0的层数；LPC<NL 时启用跨卡中转


def rel(a, b):
    d = (a.float() - b.float()).abs()
    return float(d.mean() / b.float().abs().mean().clamp_min(1e-9)), float(d.max())


def head_logits(m, h):
    """h: [1,1,HCH] -> logits [V]（用模型自己的 mixer + lm_head）"""
    mm = E.gated_residual(h, m.mix_norm, m.mix_down, m.mix_up, None)
    return E.lin(mm.view(1, E.H), m.lm_head).float().view(-1)


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/models/Qwen3.8-Flash-Next")
    text = open("/root/ascend-dev/eval_text.txt", encoding="utf-8").read()
    ids = tok(text, return_tensors="pt").input_ids[0, :NT].tolist()
    print(f"=== 管线对比 NL={NL} LPC={LPC} TOK={NT}  文本={tok.decode(ids)!r} ===", flush=True)

    m = FnModel(P=2048, lpc=LPC, dev0="npu:0", dev1="npu:1", nlayers=NL,
                with_head=True, with_ngram=False, verbose=False)
    layers = m.layers0 + m.layers1
    print(f"    层数={len(layers)}  前 {len(m.layers0)} 张卡0 / {len(m.layers1)} 张卡1", flush=True)

    # ---------- A 层主序 ----------
    with torch.no_grad():
        m.reset()
        hs = [m.emb_buf.view(1, 1, E.H).repeat(1, 1, E.HC) * 0 for _ in range(NT)]
        for t, tid in enumerate(ids):
            hs[t] = m.embed_row(tid).view(1, 1, E.H).repeat(1, 1, E.HC)
        for i, L in enumerate(layers):
            outs = []
            for t in range(NT):
                pos = torch.tensor(t, dtype=torch.int64, device=L.dev)
                outs.append(L.forward(hs[t].to(L.dev), pos, None).detach())
            hs = outs
            if i % 16 == 15:
                print(f"    [A] layer {i} done", flush=True)
        lgA = [head_logits(m, h.to(m.h0.device)).clone() for h in hs]

        # ---------- B token 主序 eager ----------
        m.reset()
        lgB = []
        for tid in ids:
            m.step(int(tid), with_head=True)
            lgB.append(m.logits.float().view(-1).clone())

        # ---------- C token 主序 + 图捕获 ----------
        m.reset()
        m.capture()
        m.reset()
        lgC = []
        for tid in ids:
            m.step(int(tid), with_head=True)
            lgC.append(m.logits.float().view(-1).clone())

    print("\n  pos | A vs B        | A vs C        | B vs C")
    for t in range(NT):
        ab = rel(lgA[t], lgB[t])[0] * 100
        ac = rel(lgA[t], lgC[t])[0] * 100
        bc = rel(lgB[t], lgC[t])[0] * 100
        print(f"   {t:2d} | {ab:8.3f}%      | {ac:8.3f}%      | {bc:8.3f}%")
    ab = rel(torch.stack(lgA), torch.stack(lgB))[0] * 100
    ac = rel(torch.stack(lgA), torch.stack(lgC))[0] * 100
    bc = rel(torch.stack(lgB), torch.stack(lgC))[0] * 100
    print(f"\n  整体 A-B {ab:.3f}%   A-C {ac:.3f}%   B-C {bc:.3f}%")
    nll = lambda L: sum(float(-torch.log_softmax(L[t], -1)[ids[t + 1]]) for t in range(NT - 1)) / (NT - 1)
    print(f"  NLL  A={nll(lgA):.4f}  B={nll(lgB):.4f}  C={nll(lgC):.4f}   ln(V)={torch.log(torch.tensor(float(E.VOCAB))).item():.4f}")
    if ab < 0.5 and bc > 2.0:
        print("  ★ 判定：A≈B 但 B≠C → **图捕获（NPUGraph replay）是 bug**")
    elif ab > 2.0:
        print("  ★ 判定：A≠B → **step()/body0/body1/跨卡中转 是 bug**")
    else:
        print("  ★ 判定：三者一致 → 浅层没差别，要加深层数（PD_NL 更大）再看")


if __name__ == "__main__":
    main()
