"""★★★★★ 终极一测：**参考实现 + 原始 bf16 权重，逐层流式跑到 48 层**，并输出 NLL-深度曲线。

要回答的只有一个问题：**这份 checkpoint 到底是不是一个可用的语言模型？**

为什么之前答不了：
- 参考实现（bf16）在 4/8 层给 12.30/12.24，和我们（QP）的 12.28/12.25 一致 —— 但浅层本来就没信息量；
- 48 层 bf16 要 ~233 GB，而 <ascend-host> 只有 201 GB 可用 → 直接装不下。

本脚本的做法：**层主序 + 逐层流式**。一次只把一层（≈5.3 GB）读进内存，喂完 T 个 token、
量一次"如果模型到这里结束"的 NLL，然后释放该层再读下一层。峰值内存只有几 GB。
副作用极好：**一次运行就得到 1..48 层的完整 NLL 曲线**，能直接看出模型是否随深度变好。

用法（容器内，服务不用停，纯 CPU）：
  cd /root/ascend-dev/fnx && /root/ascend-dev/venv/bin/python -u tests/t_ref_bf16_depth.py
环境变量：RD_NL=48 RD_TOK=8 RD_OUT=/tmp/rd.json
"""
import json
import math
import os
import time

import torch

from fnx import engine as E
from fnx.load import SRCDIR

from safetensors import safe_open                                                          # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig       # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import (                             # noqa: E402
    Qwen4ExpTextDecoderLayer, Qwen4ExpTextGatedResidual, Qwen4ExpTextRotaryEmbedding)
from transformers.cache_utils import DynamicCache                                          # noqa: E402

NL = int(os.environ.get("RD_NL", "48"))
NT = int(os.environ.get("RD_TOK", "8"))
OUT = os.environ.get("RD_OUT", "/tmp/rd.json")
DEV = "cpu"


def main():
    cfg = Qwen4ExpTextConfig(**json.load(open(os.path.join(SRCDIR, "config.json")))["text_config"])
    cfg.ple_layer_ids = []            # PLE 需要 51 GB 表，两侧都不开；已证实 PLE 无影响
    wm = json.load(open(os.path.join(SRCDIR, "model.safetensors.index.json")))["weight_map"]
    print(f"=== 参考实现 + 原始 bf16，逐层流式  NL={NL} TOK={NT} ===", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/models/Qwen3.8-Flash-Next")
    text = open("/root/ascend-dev/eval_text.txt", encoding="utf-8").read()
    ids = tok(text, return_tensors="pt").input_ids[0, :NT + 1].tolist()
    print(f"    {len(ids)} token: {tok.decode(ids)!r}", flush=True)

    # ---- 非层权重 ----
    need, byf = {}, {}
    for n in ["model.language_model.embed_tokens.weight", "lm_head.weight",
              *[k for k in wm if k.startswith("model.language_model.hyper_connection_mixer.")]]:
        byf.setdefault(wm[n], []).append(n)
    for f, ns in byf.items():
        with safe_open(os.path.join(SRCDIR, f), framework="pt", device="cpu") as fh:
            for n in ns:
                need[n.replace("model.language_model.", "")] = fh.get_tensor(n)
    emb_w = need.pop("embed_tokens.weight").to(DEV)
    lm_w = need.pop("lm_head.weight").to(DEV)
    mixer = Qwen4ExpTextGatedResidual(cfg, use_combine=False).to(DEV, dtype=torch.bfloat16).eval()
    mixer.load_state_dict({k[len("hyper_connection_mixer."):]: v for k, v in need.items()
                           if k.startswith("hyper_connection_mixer.")}, strict=False)
    rot = Qwen4ExpTextRotaryEmbedding(cfg).to(DEV)
    del need
    V = int(lm_w.shape[0])

    T = len(ids) - 1
    hs = [torch.nn.functional.embedding(torch.tensor([ids[t]]), emb_w).view(1, 1, E.H)
          .repeat(1, 1, cfg.hc_count).to(torch.bfloat16) for t in range(T)]

    def curve_nll():
        """当前 hs 走一遍 mixer + lm_head，算 teacher-forced 平均 NLL。"""
        tot = 0.0
        for t in range(T):
            lg = torch.nn.functional.linear(mixer(hs[t]).view(1, E.H), lm_w).float().view(-1)
            tot += float(-torch.log_softmax(lg, -1)[ids[t + 1]])
        return tot / T

    t0 = time.time()
    curve = []
    base = curve_nll()
    curve.append((0, base))
    print(f"   depth  0 (只有嵌入) mean_nll = {base:.4f}   (ln V = {math.log(V):.4f})", flush=True)

    for i in range(NL):
        names = [n for n in wm if n.startswith(f"model.language_model.layers.{i}.")
                 and ".ple." not in n and "ngram" not in n]
        byf = {}
        for n in names:
            byf.setdefault(wm[n], []).append(n)
        sd = {}
        for f, ns in byf.items():
            with safe_open(os.path.join(SRCDIR, f), framework="pt", device="cpu") as fh:
                for n in ns:
                    sd[n.replace(f"model.language_model.layers.{i}.", "")] = fh.get_tensor(n)
        lay = Qwen4ExpTextDecoderLayer(cfg, i).to(DEV, dtype=torch.bfloat16).eval()
        miss, _ = lay.load_state_dict(sd, strict=False)
        bad = [m for m in miss if "ngram" not in m and ".ple." not in m]
        if bad:
            print(f"   !! L{i} 缺权重 {bad[:5]}", flush=True)
        del sd

        cache = DynamicCache(config=cfg)
        outs = []
        for t in range(T):
            k = t + 1
            pid = torch.arange(k, device=DEV).view(1, 1, k)
            cos, sin = rot(hs[t], pid)
            am = torch.zeros(1, 1, 1, k, dtype=torch.bfloat16, device=DEV)
            cm = torch.ones(1, 1, dtype=torch.bool, device=DEV)
            outs.append(lay(hs[t], (cos, sin), attention_mask=am, conv_mask=cm,
                            past_key_values=cache).detach())
        hs = outs
        del lay, cache

        n = curve_nll()
        curve.append((i + 1, n))
        print(f"   depth {i+1:2d}          mean_nll = {n:.4f}   {time.time()-t0:7.1f}s", flush=True)

    print("\n=== NLL-深度曲线 ===", flush=True)
    print("  depth |  mean_nll")
    for d, n in curve:
        print(f"   {d:4d} | {n:9.4f}")
    print(f"\n  ★ 对照：我们 46 层(QP, PLE 开) = 12.6110 ；我们 48 层(QP) = 12.5084")
    print(f"  ★ 均匀随机 ln(V) = {math.log(V):.4f}")
    best = min(curve, key=lambda x: x[1])
    print(f"  ★ 最优深度 = {best[0]} 层，mean_nll = {best[1]:.4f}")
    json.dump({"nl": NL, "nt": NT, "curve": curve, "ln_v": math.log(V)}, open(OUT, "w"))
    print(f"  已写入 {OUT}", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
