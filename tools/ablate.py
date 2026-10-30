"""★ 消融定位：整机 top1=0%、NLL≈ln(V)，但单层对拍 PASS、lm_head 一致。
说明坏在一个"只有在整机拼装时才生效"的部件。本脚本加载一次小层数模型，
然后依次把某个部件**输出乘 0**，看 loss 是变好还是变坏。

判据很反直觉：某个部件如果是坏的，**去掉它应该让 loss 变好**。
所以关注点是"哪一项把 mean_nll 拉下来了"，而不是"哪一项让它变差"。

用法（容器内，不占线上服务）：
  cd /root/ascend-dev/fnx && DIAG_NL=6 /root/ascend-dev/venv/bin/python tools/ablate.py
"""
import math
import os

import torch
from transformers import AutoTokenizer

from fnx import engine as E
from fnx.model import FnModel

NL = int(os.environ.get("DIAG_NL", "6"))
LPC = int(os.environ.get("DIAG_LPC", "6"))
CAP = int(os.environ.get("DIAG_CAP", "60"))

TEXT = (
    "昇腾处理器面向大规模深度学习训练与推理场景，其软件栈以 CANN 为核心，"
    "向上提供框架适配层，向下管理芯片的算力与内存资源。在实际部署中，"
    "最容易被低估的环节往往不是算子本身的峰值性能，而是数据在设备之间搬运所带来的固定开销。"
)


def evaluate(m, ids, tag):
    """teacher-forcing 平均 NLL / top1。"""
    nll, hit, n = 0.0, 0, 0
    for i in range(len(ids) - 1):
        m.step(int(ids[i]), with_head=True)
        lg = m.logits.float().view(-1)
        tgt = ids[i + 1]
        nll += float(-torch.log_softmax(lg, dim=-1)[tgt].item())
        hit += int(int(lg.argmax().item()) == tgt)
        n += 1
    mn = nll / max(1, n)
    print(f"    {tag:22s} mean_nll={mn:8.4f}  ppl={math.exp(mn):11.2f}  top1={hit/n*100:5.2f}%",
          flush=True)
    return mn


def main():
    tok = AutoTokenizer.from_pretrained("/models/Qwen3.8-Flash-Next")
    ids = tok(TEXT, return_tensors="pt").input_ids[0].tolist()[:CAP + 1]
    print(f"=== 模型 NL={NL} LPC={LPC} 评测 {len(ids)-1} 个位置  "
          f"ln(vocab)={math.log(len(tok)):.4f} ===")

    m = FnModel(P=2048, lpc=LPC, dev0="npu:0", dev1="npu:1", nlayers=NL,
                with_head=True, verbose=False)
    layers = m.layers0 + m.layers1

    def run(tag):
        m.reset()
        return evaluate(m, ids, tag)

    base = run("baseline")

    # --- 消融 1：关掉 n-gram 特征（51 GB 磁盘表，宿主 gather，最可疑）---
    saved_table = m.table
    m.table = None
    m.reset()
    m.ngram_emb.zero_()
    run("no_ngram")
    m.table = saved_table

    # --- 消融 2：关掉 PLE 注入 ---
    orig_ple = [(L, L._ple) for L in layers if L.use_ple]
    for L, f in orig_ple:
        L._ple = (lambda f: (lambda *a, **k: f(*a, **k) * 0))(f)
    run("no_ple")
    for L, f in orig_ple:
        L._ple = f

    # --- 消融 3：关掉 MoE ---
    orig_moe = [(L, L._moe) for L in layers]
    for L, f in orig_moe:
        L._moe = (lambda f: (lambda x: f(x) * 0))(f)
    run("no_moe")
    for L, f in orig_moe:
        L._moe = f

    # --- 消融 4：关掉注意力/deltanet（token mixing）---
    orig_att = []
    for L in layers:
        nm = "_deltanet" if L.ltype in ("linear", "ple") else "_qsa"
        f = getattr(L, nm)
        orig_att.append((L, nm, f))
        setattr(L, nm, (lambda f: (lambda *a, **k: f(*a, **k) * 0))(f))
    run("no_attn")
    for L, nm, f in orig_att:
        setattr(L, nm, f)

    # --- 消融 5：MoE + 注意力都关（只留残差/shortcut 通路）---
    for L, f in orig_moe:
        L._moe = (lambda f: (lambda x: f(x) * 0))(f)
    for L, nm, f in orig_att:
        setattr(L, nm, (lambda f: (lambda *a, **k: f(*a, **k) * 0))(f))
    run("no_moe+no_attn")
    for L, f in orig_moe:
        L._moe = f
    for L, nm, f in orig_att:
        setattr(L, nm, f)

    # --- 消融 6：位置永远 0（检验 rope/位置这条腿）---
    orig_step = m.step
    p0, p1 = m.pos0, m.pos1

    def step_pos0(tok_, with_head=True):
        orig_step(tok_, with_head)
        p0.zero_()
        p1.zero_()

    m.step = step_pos0
    m.reset()
    p0.zero_()
    p1.zero_()
    evaluate(m, ids, "pos_always0")
    m.step = orig_step

    print(f"\n=== baseline mean_nll={base:.4f}；任何一项若明显低于它，那一项就是嫌疑 ===")


if __name__ == "__main__":
    with torch.no_grad():
        main()
