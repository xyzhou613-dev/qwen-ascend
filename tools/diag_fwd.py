"""★ 前向自诊断：不依赖 transformers 参考，只看"输入信号有没有走到 logits"。

背景：整机 top1=0.00%、NLL≈ln(V)（比均匀随机还差），单卡 8 层也一样。
所以不是跨卡、不是层数、不是 kernel。本脚本用极小的层数（加载快）回答三个问题：

  Q1 输入嵌入到底有没有进到残差流？（看 |emb|、|h0|，以及把 emb 置零后 logits 变不变）
  Q2 残差流每一步有没有在变化？（相邻步 h0 的余弦相似度）
  Q3 lm_head 这条腿对不对？（设备 logits vs 用同一份 QW 在宿主分块算出来的 logits）

用法（容器内，不需要起服务）：
  cd /root/ascend-dev/fnx && /root/ascend-dev/venv/bin/python tools/diag_fwd.py
"""
import os

import torch

from fnx import engine as E
from fnx.model import FnModel

NL = int(os.environ.get("DIAG_NL", "4"))
LPC = int(os.environ.get("DIAG_LPC", "4"))
IDS = [151644, 872, 198, 9707, 151645, 198]     # 任意固定 id 串，语义不重要


def show(name, t):
    tf = t.float()
    print(f"    {name:12s} shape={tuple(t.shape)} dtype={t.dtype} "
          f"|x|={tf.norm().item():.4e} max={tf.abs().max().item():.4e} "
          f"finite={bool(torch.isfinite(tf).all().item())}")


def host_logits(m, h):
    """用同一份 lm_head 的 QW 在宿主分块反量化算 logits（Q3 的参考）。"""
    lw = m.lm_head
    q, s = lw.q.cpu(), lw.s.cpu()
    V, K = q.shape
    x = h.view(-1).float().cpu()
    if x.numel() != K:
        raise RuntimeError(f"输入维度 {x.numel()} != lm_head K={K}")
    gs = K // s.shape[-1]
    xg = x.view(K // gs, gs)
    out = torch.empty(V, dtype=torch.float32)
    for n0 in range(0, V, 32768):
        n1 = min(n0 + 32768, V)
        qc = q[n0:n1].view(n1 - n0, K // gs, gs).float()
        out[n0:n1] = (torch.einsum("ngk,gk->ng", qc, xg) * s[n0:n1].float()).sum(-1)
    return out


def main():
    print(f"=== 自变量：NL={NL} LPC={LPC} FNX_PACK={os.environ.get('FNX_PACK')} "
          f"FNX_EXP_G={os.environ.get('FNX_EXP_G')} ===")
    m = FnModel(P=2048, lpc=LPC, dev0="npu:0", dev1="npu:1", nlayers=NL,
                with_head=True, verbose=False)
    m.reset()

    print("\n--- Q1：输入侧 ---")
    m.step(IDS[0], with_head=False)
    show("emb_buf", m.emb_buf)
    if m.table is not None:
        show("ngram_emb", m.ngram_emb)
    else:
        print("    ngram_emb   : table is None（ngram 特征未启用）")
    show("h0(after b0)", m.h0)
    show("h1(after b1)", m.h1)
    show("lm_head.q", m.lm_head.q)
    show("lm_head.s", m.lm_head.s)

    print("\n--- Q2：残差流随步变化 ---")
    prev = None
    for i, tid in enumerate(IDS[:5]):
        m.step(tid, with_head=True)
        h = m.h0.float().view(-1)
        lg = m.logits.float().view(-1)
        line = f"    step{i} |h0|={h.norm().item():.4e} |logits|={lg.norm().item():.4e} " \
               f"argmax={int(lg.argmax().item())} max={lg.max().item():.4f}"
        if prev is not None:
            cos = float((h * prev).sum() / (h.norm() * prev.norm() + 1e-12))
            line += f" cos(h0,h0_prev)={cos:.6f}"
        print(line)
        prev = h

    print("\n--- Q1b：把输入嵌入置零，logits 会变吗？（不变=输入根本没接上）---")
    m.reset()
    for tid in IDS[:4]:
        m.step(tid, with_head=True)
    lgA = m.logits.float().view(-1).clone()
    m.reset()
    for i, tid in enumerate(IDS[:4]):
        if i == 3:
            m.emb_buf.zero_()          # 只把最后一步的输入置零
            if m.table is not None:
                m.ngram_emb.zero_()
            if m.g0 is not None:
                m.g0.replay()
            else:
                m.body0()
            m.h_host.copy_(m.h0)
            m.h1.copy_(m.h_host)
            if m.layers1:
                if m.g1 is not None:
                    m.g1.replay()
                else:
                    m.body1()
            m.head()
        else:
            m.step(tid, with_head=True)
    lgB = m.logits.float().view(-1).clone()
    d = (lgA - lgB).abs()
    print(f"    |Δlogits| mean={d.mean().item():.6e} max={d.max().item():.6e} "
          f"rel={float(d.mean()/(lgA.abs().mean()+1e-12)):.6e}")
    print(f"    argmaxA={int(lgA.argmax().item())} argmaxB={int(lgB.argmax().item())}")

    print("\n--- Q3：lm_head 设备路径 vs 宿主分块参考 ---")
    mm = E.gated_residual(m.h0, m.mix_norm, m.mix_down, m.mix_up, None)
    show("mixer out m", mm)
    hl = host_logits(m, mm)
    lg = m.logits.float().view(-1).cpu()
    n = min(hl.numel(), lg.numel())
    dd = (hl[:n] - lg[:n]).abs()
    print(f"    设备 |logits|={lg.norm().item():.4e}  宿主 |logits|={hl.norm().item():.4e}")
    print(f"    |Δ| mean={dd.mean().item():.6e} max={dd.max().item():.6e} "
          f"rel={float(dd.mean()/(hl.abs().mean()+1e-12)):.6e}")
    print(f"    argmax 设备={int(lg.argmax().item())} 宿主={int(hl.argmax().item())}")
    print(f"    top5 设备={lg.topk(5).indices.tolist()}")
    print(f"    top5 宿主={hl.topk(5).indices.tolist()}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
