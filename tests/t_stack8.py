"""★★★ 8 层全栈端到端对照（逐 token + DynamicCache）：参考实现 vs 我们的实现。

为什么是这一测：所有 pos=0 的对拍都过了、所有消融都无效、PLE/量化/缺层都排除。
唯一没被验证的是 **pos>0 且跨多层的串行路径**（KV 缓存 + 每层写回）。
本脚本让两侧都用"逐 token + 保留历史"的方式跑同一段真实文本，比较
  (a) 每个位置的 logits 相对误差
  (b) 各自的 teacher-forced NLL —— **参考实现自己是多少，这才是有意义的基准**

两侧都把 PLE 关掉（参考侧 51 GB ngram 表无法实例化）。

用法（容器内，先停服）： cd /root/ascend-dev/fnx && python -u tests/t_stack8.py
"""
import json
import math
import os
import sys

import torch

from fnx import engine as E
from fnx.load import QDIR, SRCDIR
from fnx.model import FnModel

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig       # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import (                              # noqa: E402
    Qwen4ExpTextDecoderLayer, Qwen4ExpTextGatedResidual, Qwen4ExpTextRotaryEmbedding)
from transformers.cache_utils import DynamicCache                                            # noqa: E402

DEV = "npu:0"
# 参考侧默认放 CPU：dav-c220 上参考 MoE 的 aclnnNonzero 会稳定报 vector core 异常(507035)，
# 而宿主有 218 GB 内存，4~8 层 bf16（20~40 GB）完全放得下。
RDEV = "cpu" if os.environ.get("ST_REF_CPU", "1") == "1" else DEV
NL = int(os.environ.get("ST_NL", "8"))
NT = int(os.environ.get("ST_TOK", "24"))


def rel(a, b):
    d = (a.float() - b.float()).abs()
    return float(d.mean() / b.float().abs().mean().clamp_min(1e-9))


def main():
    cfg = Qwen4ExpTextConfig(**json.load(open(os.path.join(SRCDIR, "config.json")))["text_config"])
    cfg.ple_layer_ids = []
    print(f"=== 8 层全栈对照 NL={NL} NT={NT} hc={cfg.hc_count} ===", flush=True)

    # ---- 文本 ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/models/Qwen3.8-Flash-Next")
    text = open("/root/ascend-dev/eval_text.txt", encoding="utf-8").read()
    ids = tok(text, return_tensors="pt").input_ids[0, :NT + 1].tolist()
    print(f"    {NT+1} 个 token: {tok.decode(ids)[:40]!r}", flush=True)

    # ---- 参考侧权重 ----
    from safetensors import safe_open
    wm = json.load(open(os.path.join(SRCDIR, "model.safetensors.index.json")))["weight_map"]
    keys = ["model.language_model.embed_tokens.weight", "lm_head.weight"]
    for i in range(NL):
        keys += [n for n in wm if n.startswith(f"model.language_model.layers.{i}.")
                 and ".ple." not in n and "ngram" not in n]
    keys += [n for n in wm if n.startswith("model.language_model.hyper_connection_mixer.")]
    byf, need = {}, {}
    for n in keys:
        byf.setdefault(wm[n], []).append(n)
    for f, ns in byf.items():
        with safe_open(os.path.join(SRCDIR, f), framework="pt", device="cpu") as fh:
            for n in ns:
                need[n.replace("model.language_model.", "")] = fh.get_tensor(n)
    print(f"   [1/3] 读到 bf16 键 {len(need)} 个，开始上卡...", flush=True)

    emb_w = need.pop("embed_tokens.weight").to(RDEV)
    lm_w = need.pop("lm_head.weight").to(RDEV)
    mixer = Qwen4ExpTextGatedResidual(cfg, use_combine=False).to(RDEV, dtype=torch.bfloat16).eval()
    mixer.load_state_dict({k[len("hyper_connection_mixer."):]: v for k, v in need.items()
                           if k.startswith("hyper_connection_mixer.")}, strict=False)
    refs = []
    for i in range(NL):
        pre = f"layers.{i}."
        lay = Qwen4ExpTextDecoderLayer(cfg, i).to(RDEV, dtype=torch.bfloat16).eval()
        m2, _ = lay.load_state_dict({k[len(pre):]: v for k, v in need.items() if k.startswith(pre)},
                                    strict=False)
        bad = [x for x in m2 if "ngram" not in x and ".ple." not in x]
        if bad:
            print(f"   !! 参考 L{i} 缺权重 {bad[:4]}", flush=True)
        refs.append(lay)
    del need
    rot = Qwen4ExpTextRotaryEmbedding(cfg).to(RDEV)
    print(f"   [2/3] 参考 {NL} 层就绪，开始逐 token...", flush=True)

    lg_ref, lg_our = [], []
    with torch.no_grad():
        cache = DynamicCache(config=cfg)      # ★ 必须按 config 初始化，否则 layers[] 越界
        hs = torch.zeros(1, 1, E.HCH, dtype=torch.bfloat16, device=RDEV)
        for t in range(len(ids) - 1):
            e = torch.nn.functional.embedding(
                torch.tensor([ids[t]], device=RDEV), emb_w).view(1, 1, E.H)
            hs = e.repeat(1, 1, cfg.hc_count)
            k = t + 1
            # ★ 必须给**全部位置**的 cos/sin：QSA 的 indexer 要按完整位置取（它会自己切到当前位），
            # 只给当前位置会让 indexer 里 full_cos[...] 越界。
            pid = torch.arange(k, device=RDEV).view(1, 1, k)
            cos, sin = rot(hs, pid)
            am = torch.zeros(1, 1, 1, k, dtype=torch.bfloat16, device=RDEV)
            cm = torch.ones(1, 1, dtype=torch.bool, device=RDEV)
            for L in refs:
                hs = L(hs, (cos, sin), attention_mask=am, conv_mask=cm,
                       past_key_values=cache)
            lg_ref.append(torch.nn.functional.linear(mixer(hs).view(1, E.H), lm_w).float().view(-1).clone())
    print(f"   [3/3] 参考前向完成，跑我们这一侧...", flush=True)

    m = FnModel(P=2048, lpc=NL, dev0=DEV, dev1="npu:1", nlayers=NL,
                with_head=True, with_ngram=False, verbose=False)
    for L in m.layers0 + m.layers1:
        L.use_ple = False
    with torch.no_grad():
        m.reset()
        for t in range(len(ids) - 1):
            m.step(int(ids[t]), with_head=True)
            lg_our.append(m.logits.float().view(-1).cpu().clone())

    nll_r = sum(-torch.log_softmax(lg_ref[t], -1)[ids[t + 1]].item() for t in range(len(lg_ref)))
    nll_o = sum(-torch.log_softmax(lg_our[t], -1)[ids[t + 1]].item() for t in range(len(lg_our)))
    n = len(lg_ref)
    print("\n  pos | 参考 NLL | 我们 NLL | logits rel")
    for t in range(min(n, 10)):
        print(f"   {t:2d} | {-(torch.log_softmax(lg_ref[t],-1)[ids[t+1]].item()):8.4f} "
              f"| {-(torch.log_softmax(lg_our[t],-1)[ids[t+1]].item()):8.4f} "
              f"| {rel(lg_our[t], lg_ref[t])*100:7.3f}%")
    print(f"\n  ★ 参考 {NL} 层 mean_nll = {nll_r/n:.4f}")
    print(f"  ★ 我们 {NL} 层 mean_nll = {nll_o/n:.4f}")
    print(f"  ★ ln(V) = {math.log(int(lg_ref[0].numel())):.4f}")
    print(f"  ★ 整体 logits rel = {rel(torch.stack(lg_our), torch.stack(lg_ref))*100:.3f}%")


if __name__ == "__main__":
    with torch.no_grad():
        main()