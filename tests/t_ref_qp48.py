"""★★★★ 决定性实验：**参考实现 + 我们同样的 QP 权重**，跑到 48 层。

为什么这是唯一还能定案的实验：
- 参考实现（bf16 原始权重）在 4 层 / 8 层给出 12.30 / 12.24，和我们（QP 权重）的
  12.28 / 12.25 一致 → 说明**浅层**上两边等价，但也说明浅层本来就没信息量。
- 我们 48 层（QP）给 12.51。要判定"是权重坏了还是我们的代码在深层坏了"，
  必须让**同一个实现栈、同样的权重、同样的深度**只差"谁写的代码"。
- 本机跑不了 bf16 48 层（233 GB > 201 GB 可用），但 QP 形式只要 ~2.9 GB/层，
  且可以**逐层反量化、逐层释放**（峰值十几 GB）。

做法：**层主序**（layer-major）。对第 i 层：加载 QP → 反量化成 bf16 → 装进参考模块 →
用它自己的 DynamicCache 逐 token 喂完 T 个位置 → 把输出作为下一层的输入 → 释放该层。
数学上等价于 token 主序（因果模型里第 i 层第 t 个位置只依赖 ≤t）。

用法（容器内，先停服）：
  cd /root/ascend-dev/fnx && FNX_EXP_G=64 python -u tests/t_ref_qp48.py
环境变量：RQ_NL=48 RQ_TOK=8 RQ_OUT=/tmp/rq.json
"""
import json
import os
import time

import torch

from fnx import engine as E
from fnx.load import QDIR, SRCDIR, load_layer

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig      # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import (                            # noqa: E402
    Qwen4ExpTextDecoderLayer, Qwen4ExpTextGatedResidual, Qwen4ExpTextRotaryEmbedding)
from transformers.cache_utils import DynamicCache                                         # noqa: E402

NL = int(os.environ.get("RQ_NL", "48"))
NT = int(os.environ.get("RQ_TOK", "8"))
OUT = os.environ.get("RQ_OUT", "/tmp/rq.json")
DEV = "cpu"                      # 参考实现放 CPU：NPU 上参考 MoE 的 nonzero 会 507035


def plain(w, K=None):
    """把 QP 侧的各种包装器/量化权重变成普通 bf16 张量（参考模块只认 nn.Parameter）。"""
    if isinstance(w, E.Raw):
        return w.w.to(torch.bfloat16)
    if isinstance(w, E.QW):
        d = E.deq(w.q, w.s)
        return (d[..., :K] if K else d).to(torch.bfloat16)
    if isinstance(w, E.ExpQ):
        n = w.q.shape[0]
        return w.gather_deq(torch.arange(n)).to(torch.bfloat16)
    return w.to(torch.bfloat16)


def main():
    cfg = Qwen4ExpTextConfig(**json.load(open(os.path.join(SRCDIR, "config.json")))["text_config"])
    cfg.ple_layer_ids = []
    print(f"=== 参考实现 + QP 权重：NL={NL} NT={NT} hc={cfg.hc_count} ===", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/models/Qwen3.8-Flash-Next")
    text = open("/root/ascend-dev/eval_text.txt", encoding="utf-8").read()
    ids = tok(text, return_tensors="pt").input_ids[0, :NT + 1].tolist()
    print(f"    {len(ids)} token: {tok.decode(ids)!r}", flush=True)

    # 非层权重：embed / mixer / lm_head（bf16 原件，和参考侧一致）
    from safetensors import safe_open
    wm = json.load(open(os.path.join(SRCDIR, "model.safetensors.index.json")))["weight_map"]
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
    mm, mu = mixer.load_state_dict(
        {k[len("hyper_connection_mixer."):]: v for k, v in need.items()
         if k.startswith("hyper_connection_mixer.")}, strict=False)
    print(f"   mixer missing={list(mm)} unexpected={list(mu)}", flush=True)
    rot = Qwen4ExpTextRotaryEmbedding(cfg).to(DEV)

    T = len(ids) - 1
    hs = [torch.nn.functional.embedding(torch.tensor([ids[t]]), emb_w).view(1, 1, E.H).repeat(
        1, 1, cfg.hc_count).to(torch.bfloat16) for t in range(T)]
    hs = [h.to(DEV) for h in hs]

    def curve_nll():
        tot = 0.0
        for t in range(T):
            lg = torch.nn.functional.linear(mixer(hs[t]).view(1, E.H), lm_w).float().view(-1)
            tot += float(-torch.log_softmax(lg, -1)[ids[t + 1]])
        return tot / T

    t0 = time.time()
    curve = [(0, curve_nll())]
    print(f"   depth  0 (只有嵌入) mean_nll = {curve[0][1]:.4f}", flush=True)

    for i in range(NL):
        lay = Qwen4ExpTextDecoderLayer(cfg, i).to(DEV, dtype=torch.bfloat16).eval()
        W = load_layer(os.path.join(QDIR, f"L{i:02d}.safetensors"), i, DEV)
        sd = {}
        for k, v in W.items():
            if k.startswith("mlp.experts."):
                continue
            K = None
            if isinstance(v, E.QW):
                K = v.K
            sd[k] = plain(v, K)
        sd["mlp.experts.gate_up_proj"] = plain(W["mlp.experts.gate_up_proj"])
        sd["mlp.experts.down_proj"] = plain(W["mlp.experts.down_proj"])
        del W
        miss, unexp = lay.load_state_dict(sd, strict=False)
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
            o = lay(hs[t], (cos, sin), attention_mask=am, conv_mask=cm, past_key_values=cache)
            outs.append(o.detach())
        hs = outs
        del lay, cache
        n = curve_nll()
        curve.append((i + 1, n))
        print(f"   depth {i+1:2d}          mean_nll = {n:.4f}   {time.time()-t0:7.1f}s", flush=True)

    lg = [torch.nn.functional.linear(mixer(hs[t]).view(1, E.H), lm_w).float().view(-1) for t in range(T)]
    nll = [float(-torch.log_softmax(lg[t], -1)[ids[t + 1]]) for t in range(T)]
    mean = sum(nll) / len(nll)
    import math
    print("\n  pos | NLL")
    for t in range(T):
        print(f"   {t:2d} | {nll[t]:8.4f}")
    print(f"\n  ★★ 参考实现 + QP 权重，{NL} 层 mean_nll = {mean:.4f}")
    print(f"  ★★ ln(V) = {math.log(int(lg[0].numel())):.4f}")
    print(f"  ★★ 对照：参考+bf16 48 层 = 5.4437 ；我们(QP,kernel) 48 层 = 12.5084")
    best = min(curve, key=lambda x: x[1])
    print(f"  ★★ 本曲线最优深度 = {best[0]} 层，mean_nll = {best[1]:.4f}")
    json.dump({"nl": NL, "nt": NT, "curve": curve, "mean_nll": mean, "nll": nll},
              open(OUT, "w"))
    print(f"  已写入 {OUT}", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
