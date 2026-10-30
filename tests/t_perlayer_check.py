"""★ 逐层图（dev1）的收益与正确性：直接走 m.step()，eager vs 逐层图 对比。

顺序：同一个模型先在 capture 之前跑 eager（g0/g1 都没有），记录 logits 与耗时；
再 capture()，reset() 后跑同样步骤，比较 logits 是否一致 + 每步快多少。
"""
import os
import time

import torch

from fnx import engine as E
from fnx.model import FnModel

NL = int(os.environ.get("PL_NL", "8"))
LPC = int(os.environ.get("PL_LPC", "4"))
N = int(os.environ.get("PL_STEPS", "10"))


def run(m, ids, tag):
    m.reset()
    lgs = []
    t0 = time.perf_counter()
    torch.npu.synchronize()
    for i in range(N):
        m.step(int(ids[i]), with_head=True)
        lgs.append(m.logits.float().view(-1).clone())
    torch.npu.synchronize()
    dt = (time.perf_counter() - t0) / N * 1e3
    print(f"    {tag:14s} {dt:8.2f} ms/step", flush=True)
    return lgs, dt


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/models/Qwen3.8-Flash-Next")
    ids = tok("昇腾处理器面向大规模深度学习训练与推理场景，其软件栈", return_tensors="pt").input_ids[0, :N].tolist()

    m = FnModel(P=2048, lpc=LPC, dev0="npu:0", dev1="npu:1", nlayers=NL,
                with_head=True, with_ngram=False, verbose=False)
    print(f"=== 逐层图验证 NL={NL} LPC={LPC}  dev0={len(m.layers0)} dev1={len(m.layers1)} "
          f"FNX_GRAPH_G1={os.environ.get('FNX_GRAPH_G1')} ===", flush=True)

    with torch.no_grad():
        # 预热
        for i in range(3):
            m.step(int(ids[i]), with_head=True)
        lg_e, t_e = run(m, ids, "eager")

        m.capture()
        print(f"    逐层图：{0 if not m._g1_list else len(m._g1_list)} 张；"
              f"g0={'有' if m.g0 is not None else '无'} g1={'有' if m.g1 is not None else '无'}", flush=True)
        lg_g, t_g = run(m, ids, "逐层图")

    d = (torch.stack(lg_e) - torch.stack(lg_g)).abs()
    rel = float(d.mean() / torch.stack(lg_e).abs().mean().clamp_min(1e-12))
    print(f"\n  logits 相对差 {rel*100:.4f}%  max {float(d.max()):.4e} "
          f"-> {'一致 ✅' if rel < 1e-3 else '不一致 ❌'}")
    print(f"  速度 {t_e:.2f} -> {t_g:.2f} ms/step （{t_e/t_g:.2f}x）")
    print(f"  外推 46 层(24/22)：eager ~{t_e*22/4 + t_e:.0f} ms  vs  逐层图 ~{t_g*22/4 + t_e:.0f} ms")


if __name__ == "__main__":
    main()
