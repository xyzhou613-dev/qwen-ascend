"""★★★ 我们自己的 NLL-深度曲线（与我们同一套 FnLayer 代码、同一批 QP 权重）。

为什么需要：参考实现 + QP 权重在 48 层拿到 5.6706，我们只有 12.5084（差 6.84 nats）→
**bug 在我们代码里**。接下来必须知道"从哪一层开始跑偏"，而不是继续猜。
本脚本用**层主序**（一次一层，NPU 峰值几 GB），每层都用测试自己的状态（KV/S/conv），
逐 token 推进 pos=0..T-1，每层后量一次"如果模型到这里结束"的 NLL。

口径与 `t_ref_bf16_depth.py` / `t_ref_qp48.py` 完全一致：同一份 eval_text 的前 T+1 个 token、
同样 PLE 关、同样 G=64。所以三条曲线可以直接叠在一起看。

用法（容器内，可与服务共存）：
  cd /root/ascend-dev/fnx && FNX_EXP_G=64 FNX_NO_PLE=1 python -u tests/t_our_depth.py
环境变量：OD_NL=48 OD_TOK=3 OD_OUT=/tmp/od48.json
"""
import json
import math
import os
import time

import torch

from fnx import engine as E
from fnx.load import QDIR, SRCDIR, load_layer, load_nonlayer

NL = int(os.environ.get("OD_NL", "48"))
NT = int(os.environ.get("OD_TOK", "3"))
OUT = os.environ.get("OD_OUT", "/tmp/od48.json")
DEV = "npu:0"


def main():
    cfg = json.load(open(os.path.join(SRCDIR, "config.json")))["text_config"]
    lt = cfg["layer_types"]
    H = E.H
    print(f"=== 我们的实现：NLL-深度曲线 NL={NL} TOK={NT} PLE={os.environ.get('FNX_NO_PLE')} ===",
          flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/models/Qwen3.8-Flash-Next")
    text = open("/root/ascend-dev/eval_text.txt", encoding="utf-8").read()
    ids = tok(text, return_tensors="pt").input_ids[0, :NT + 1].tolist()
    print(f"    {len(ids)} token: {tok.decode(ids)!r}", flush=True)

    (eq, es), lm_head, mx = load_nonlayer(DEV, DEV, with_lmhead=True)
    w_mix = mx["hc_norm.weight"]
    w_dn = mx["input_mix_weight_down.weight"]
    w_up = mx["input_mix_weight_up.weight"]

    def embed_row(t):
        e = eq[t].view(1, H).to(DEV).to(torch.bfloat16)
        s = es[t].view(1, -1).to(DEV).to(torch.bfloat16)
        gs = H // s.shape[-1]
        return (e.view(1, -1, gs) * s.unsqueeze(-1)).reshape(1, 1, H)

    T = len(ids) - 1
    hs = [embed_row(ids[t]).repeat(1, 1, E.HC) for t in range(T)]

    def curve_nll():
        tot = 0.0
        for t in range(T):
            m = E.gated_residual(hs[t], w_mix, w_dn, w_up, None)
            lg = E.lin(m.view(1, H), lm_head).float().view(-1)
            tot += float(-torch.log_softmax(lg, -1)[ids[t + 1]])
        return tot / T

    t0 = time.time()
    curve = [(0, curve_nll())]
    print(f"   depth  0 (只有嵌入) mean_nll = {curve[0][1]:.4f}", flush=True)

    for i in range(NL):
        ltype = "full" if lt[i] in E.FULL_TYPES else ("ple" if i == E.PLE_LAYER else "linear")
        W = load_layer(os.path.join(QDIR, f"L{i:02d}.safetensors"), i, DEV)
        lay = E.FnLayer(i, ltype, W, DEV)
        if ltype == "full":
            lay.kv = E.KVStore(2048, DEV, 1)
            lay.ki = 0
        outs = []
        for t in range(T):
            pos = torch.tensor(t, dtype=torch.int64, device=DEV)
            outs.append(lay.forward(hs[t], pos, None).detach())
        hs = outs
        del lay, W
        n = curve_nll()
        curve.append((i + 1, n))
        print(f"   depth {i+1:2d}          mean_nll = {n:.4f}   {time.time()-t0:7.1f}s", flush=True)

    print("\n=== 三条曲线对照（同一口径 T=%d）===" % NT, flush=True)
    ref_bf16 = {0: 11.9664, 1: 11.8885, 2: 11.5522, 4: 12.0020, 8: 11.9009,
                14: 11.6956, 20: 11.8194, 24: 12.1497, 28: 12.2447, 32: 12.3617,
                36: 12.0780, 38: 11.9034, 39: 10.9638, 40: 10.4113, 42: 9.8035,
                44: 8.8721, 45: 7.9133, 46: 8.6180, 47: 8.0754, 48: 5.4437}
    ref_qp = {0: 11.9664, 1: 11.8972, 2: 11.5418, 4: 12.0530, 8: 11.9344,
              14: 11.6930, 20: 11.9590, 24: 12.2023, 28: 12.4430, 32: 12.5663,
              36: 12.3396, 38: 11.9843, 39: 11.0493, 40: 10.5941, 42: 10.0337,
              44: 9.0267, 45: 8.0791, 46: 9.8096, 47: 8.7870, 48: 5.6706}
    print("  depth |   参考bf16 |   参考QP   |    我们    | 我们-参考QP")
    mine = dict(curve)
    for d in sorted(set(list(mine) + list(ref_qp))):
        rq = ref_qp.get(d)
        rb = ref_bf16.get(d)
        mv = mine.get(d)
        print(f"   {d:4d} | {rb if rb is None else f'{rb:9.4f}'} | "
              f"{rq if rq is None else f'{rq:9.4f}'} | "
              f"{mv if mv is None else f'{mv:9.4f}'} | "
              f"{'' if (rq is None or mv is None) else f'{mv-rq:+9.4f}'}")
    print(f"\n  ★ ln(V) = {math.log(E.VOCAB):.4f}")
    bad = [(d, mine[d] - ref_qp[d]) for d in sorted(mine)
           if d in ref_qp and mine[d] - ref_qp[d] > 0.3]
    print(f"  ★ 偏差首次 >0.3 nats 的深度：{bad[:6] if bad else '无'}")
    json.dump({"nl": NL, "nt": NT, "curve": curve}, open(OUT, "w"))
    print(f"  已写入 {OUT}", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
