"""★ 参考实现（原始 bf16、CPU、层主序流式）在同一段 eval_text 上算出**逐 token 的 NLL**，
用来回答"我们的 W4 到底离 bf16 多远"。口径与 `/fnx/loss` 逐位对齐（同一文本、同一 token 前缀）。

与 t_ref_bf16_depth.py 的区别：只关心**最后一层深度**的逐 token NLL + 汇总（mean/top1/median_rank），
并输出 JSON 便于和线上服务做配对比较。

用法（容器内，纯 CPU，服务不用停）：
  RD_TOK=12 RD_NL=48 RD_OUT=/tmp/ref_eval12.json \
    /root/ascend-dev/venv/bin/python -u tests/t_ref_eval.py
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
NT = int(os.environ.get("RD_TOK", "12"))
OUT = os.environ.get("RD_OUT", "/tmp/ref_eval.json")
DEV = "cpu"


def main():
    cfg = Qwen4ExpTextConfig(**json.load(open(os.path.join(SRCDIR, "config.json")))["text_config"])
    cfg.ple_layer_ids = []
    wm = json.load(open(os.path.join(SRCDIR, "model.safetensors.index.json")))["weight_map"]
    print(f"=== 参考 bf16 逐 token NLL  NL={NL} TOK={NT} ===", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/models/Qwen3.8-Flash-Next")
    text = open("/root/ascend-dev/eval_text.txt", encoding="utf-8").read()
    ids = tok(text, return_tensors="pt").input_ids[0, :NT + 1].tolist()
    print(f"    {len(ids)} token: {tok.decode(ids)!r}", flush=True)

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

    def per_token():
        """返回逐 token NLL、argmax 命中、目标排名。"""
        nll, hit, rank = [], 0, []
        for t in range(T):
            lg = torch.nn.functional.linear(mixer(hs[t]).view(1, E.H), lm_w).float().view(-1)
            lp = torch.log_softmax(lg, -1)
            tgt = ids[t + 1]
            nll.append(float(-lp[tgt]))
            am = int(lg.argmax())
            hit += int(am == tgt)
            rank.append(int((lg > lg[tgt]).sum()))
        return nll, hit, rank

    t0 = time.time()
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
        nll_now, hit_now, _ = per_token()
        print(f"   L{i:2d}  mean_nll={sum(nll_now)/T:.4f}  {time.time()-t0:7.1f}s", flush=True)

    nll, hit, rank = per_token()
    mean_nll = sum(nll) / T
    print(f"\n=== 参考 bf16 最终（{NL} 层、{T} 个预测）===", flush=True)
    print(f"  mean_nll = {mean_nll:.4f}   ppl = {math.exp(mean_nll):.3f}   "
          f"top1 = {hit/T*100:.2f}%   median_rank = {sorted(rank)[len(rank)//2]}", flush=True)
    print(f"  逐 token NLL: " + " ".join(f"{x:.3f}" for x in nll), flush=True)
    print(f"  逐 token rank: " + " ".join(str(x) for x in rank), flush=True)
    json.dump({"nl": NL, "nt": NT, "ids": ids, "nll": nll, "rank": rank,
               "mean_nll": mean_nll, "hit": hit, "ln_v": math.log(V)},
              open(OUT, "w"))
    print(f"  已写入 {OUT}", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
