"""★★ 逐级对拍（T=1, pos=0）：把"入口 → 层0 → 层1 → mixer → lm_head"每一级都跟参考实现比。

背景：t_e2e.py 已经确认整机对拍差 409%（参考能解出合理文字，我们解出乱码）。
但那一步只能说明"有差异"，不能说明差在哪一级。T=1/pos=0 时因果掩码退化成平凡、
KV/递归状态全是初值，参考实现不需要任何 cache，所以可以逐级严格对比。

用法（容器内，先停服）：
  cd /root/ascend-dev/fnx && /root/ascend-dev/venv/bin/python -u tests/t_e2e1.py
"""
import json
import os

import torch

from fnx import engine as E
from fnx.load import QDIR, SRCDIR, load_layer

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig      # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import (                             # noqa: E402
    Qwen4ExpTextDecoderLayer, Qwen4ExpTextGatedResidual, Qwen4ExpTextRotaryEmbedding)

DEV = "npu:0"
TOK = 108095          # "昇腾"


def rel(a, b):
    d = (a.float() - b.float()).abs()
    return float(d.mean() / b.float().abs().mean().clamp_min(1e-9)), float(d.max())


def show(tag, ours, ref):
    r, mx = rel(ours, ref)
    flag = "OK " if r < 5e-2 else ">>> 偏了"
    print(f"  {tag:22s} |我|={ours.float().norm().item():10.4e} "
          f"|参|={ref.float().norm().item():10.4e}  rel {r*100:8.3f}%  max_abs {mx:.4e}  {flag}",
          flush=True)
    return r


def main():
    cfg = Qwen4ExpTextConfig(**json.load(open(os.path.join(SRCDIR, "config.json")))["text_config"])
    cfg.ple_layer_ids = []                       # 不构造 51 GB ngram 表（会挂死）
    print(f"=== 逐级对拍 T=1 pos=0  tok={TOK}  hc={cfg.hc_count} H={cfg.hidden_size} ===", flush=True)

    # ---------- 参考：权重从 bf16 原件拿，只取需要的键 ----------
    wm = json.load(open(os.path.join(SRCDIR, "model.safetensors.index.json")))["weight_map"]
    from safetensors import safe_open
    need = {}
    keys = ["model.language_model.embed_tokens.weight", "lm_head.weight"]
    for i in (0, 1):
        keys += [n for n in wm if n.startswith(f"model.language_model.layers.{i}.")
                 and "ngram_embedding" not in n and ".ple." not in n]
    keys += [n for n in wm if n.startswith("model.language_model.hyper_connection_mixer.")]
    byf = {}
    for n in keys:
        byf.setdefault(wm[n], []).append(n)
    for f, ns in byf.items():
        with safe_open(os.path.join(SRCDIR, f), framework="pt", device="cpu") as fh:
            for n in ns:
                need[n.replace("model.language_model.", "")] = fh.get_tensor(n)
    print(f"  读到 bf16 键 {len(need)} 个", flush=True)

    ref_emb_w = need.pop("embed_tokens.weight").to(DEV)
    ref_lm_w = need.pop("lm_head.weight").to(DEV)
    ref_mixer = Qwen4ExpTextGatedResidual(cfg, use_combine=False).to(DEV, dtype=torch.bfloat16).eval()
    mm, mu = ref_mixer.load_state_dict(
        {k[len("hyper_connection_mixer."):]: v for k, v in need.items()
         if k.startswith("hyper_connection_mixer.")}, strict=False)
    print(f"  mixer load missing={list(mm)} unexpected={list(mu)}", flush=True)
    ref_L = []
    for i in (0, 1):
        pre = f"layers.{i}."
        sub = {k[len(pre):]: v for k, v in need.items() if k.startswith(pre)}
        lay = Qwen4ExpTextDecoderLayer(cfg, i).to(DEV, dtype=torch.bfloat16).eval()
        m2, u2 = lay.load_state_dict(sub, strict=False)
        miss2 = [x for x in m2 if "ngram" not in x and ".ple." not in x]
        print(f"  参考 L{i} load missing={miss2[:6]} unexpected={list(u2)[:4]}", flush=True)
        ref_L.append(lay)
    rot = Qwen4ExpTextRotaryEmbedding(cfg).to(DEV)

    # ---------- 我们：QP 权重 ----------
    ot = torch.tensor([TOK], device=DEV)
    with torch.no_grad():
        # 1) 入口嵌入
        ref_e = torch.nn.functional.embedding(ot, ref_emb_w).view(1, 1, E.H)   # [1,1,H]
        emb_q, emb_s = None, None
        from fnx.model import FnModel  # 仅借用 embed_row 的取法
        # 直接用 FnModel 的 embed_row 逻辑（避免装配整模型）
        from fnx.load import load_nonlayer
        (eq, es), lw, mx = load_nonlayer(DEV, DEV, with_lmhead=True)
        e_row = eq[TOK].view(1, E.H).to(DEV).to(torch.bfloat16)
        s_row = es[TOK].view(1, -1).to(DEV).to(torch.bfloat16)
        gs = E.H // s_row.shape[-1]
        our_e = (e_row.view(1, -1, gs) * s_row.unsqueeze(-1)).reshape(1, 1, E.H)
        show("入口嵌入", our_e, ref_e)

        hc = cfg.hc_count
        ref_h = ref_e.repeat(1, 1, hc)
        our_h = our_e.repeat(1, 1, hc)
        show("入口 repeat(hc)", our_h, ref_h)

        # 2) 逐层
        pid = torch.zeros(1, 1, 1, dtype=torch.long, device=DEV)
        cos, sin = rot(ref_h, pid)
        mask = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16, device=DEV)
        cmask = torch.ones(1, 1, dtype=torch.bool, device=DEV)
        pos = torch.zeros((), dtype=torch.int64, device=DEV)
        for i in (0, 1):
            ltype = "full" if cfg.layer_types[i] in E.FULL_TYPES else (
                "ple" if i == E.PLE_LAYER else "linear")
            W = load_layer(os.path.join(QDIR, f"L{i:02d}.safetensors"), i, DEV)
            lay = E.FnLayer(i, ltype, W, DEV)
            lay.use_ple = False                                  # 与参考一致
            if ltype == "full":
                lay.kv = E.KVStore(2048, DEV, 0)
            ref_h = ref_L[i](ref_h, (cos, sin), attention_mask=mask, conv_mask=cmask,
                             past_key_values=None)
            our_h = lay.forward(our_h, pos, None)
            show(f"第 {i} 层输出", our_h, ref_h)
            del W, lay

        # 3) mixer
        ref_o = ref_mixer(ref_h)
        our_o = E.gated_residual(our_h, mx["hc_norm.weight"], mx["input_mix_weight_down.weight"],
                                 mx["input_mix_weight_up.weight"], None)
        show("hyper_connection_mixer", our_o, ref_o)

        # 4) lm_head
        ref_lg = torch.nn.functional.linear(ref_o.view(1, E.H).to(torch.bfloat16), ref_lm_w)
        our_lg = E.lin(our_o.view(1, E.H), lw)
        show("lm_head logits", our_lg, ref_lg)

        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("/models/Qwen3.8-Flash-Next")
        print(f"\n  参考 top5 = {[tok.decode([int(x)]) for x in ref_lg[0].topk(5).indices]}")
        print(f"  我们 top5 = {[tok.decode([int(x)]) for x in our_lg[0].topk(5).indices]}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
