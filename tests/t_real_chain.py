"""真实依赖链下的 MoE 专家段耗时：24 层串联（每层 gate_up -> act -> down -> 加权求和）。

与 t_batch_bench 的区别：那里 24 层用同一份张量（只测过桥开销，且不可实现）；
这里每层的输入依赖上一层的输出（真 decode 的形态），所以数字可以直接用。
"""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from fnx.w4gemv_op import W4GemvOp, GS

GOLD = "/root/ascend-dev/golden/w4gemv"
NL = 24          # 每卡层数
TOPK = 10


def load(name):
    meta = json.load(open(GOLD + "/meta.json"))
    c = meta["cases"]["multi_" + name + "_proj"]
    d = GOLD + "/multi"
    nE, N, K = c["B"], c["N"], c["K"]
    p4 = np.fromfile(f"{d}/{name}_proj_p4.bin", dtype=np.int8).reshape(nE, N, K // 2)
    s4 = np.fromfile(f"{d}/{name}_proj_s4.bin", dtype=np.float16).reshape(nE, N, K // GS)
    return p4, s4, nE, N, K


def main():
    op = W4GemvOp()
    gu_p4, gu_s4, nE, Ngu, Kgu = load("gate_up")
    dn_p4, dn_s4, _, Ndn, Kdn = load("down")
    gu_p4 = torch.from_numpy(np.ascontiguousarray(op.to_v2_layout(gu_p4, Ngu, Kgu))).npu()
    dn_p4 = torch.from_numpy(np.ascontiguousarray(op.to_v2_layout(dn_p4, Ndn, Kdn))).npu()
    gu_s4t = op.prepare_layer(gu_s4)
    dn_s4t = op.prepare_layer(dn_s4)
    del gu_s4, dn_s4
    w = torch.rand(TOPK, device="npu:0")

    # 第一层的输入 x（真链路里来自上一个 op）；后面的 x 就是上一层 down 的输出
    x = torch.randn(TOPK, Kgu, dtype=torch.float32, device="npu:0")
    out_gu = torch.empty((TOPK, Ngu), dtype=torch.float32, device="npu:0")
    out_dn = torch.empty((TOPK, Ndn), dtype=torch.float32, device="npu:0")

    def one_layer(xin):
        y = op.run(gu_p4, gu_s4t, xin, nE, Ngu, Kgu, lay=2, out=out_gu)
        act = (F.silu(y[:, :Ngu // 2]) * y[:, Ngu // 2:]).contiguous()   # [10,640]
        y2 = op.run(dn_p4, dn_s4t, act, nE, Ndn, Kdn, lay=2, out=out_dn)
        return op.weighted_sum(y2, w.cpu().numpy())        # [2560]

    # warmup
    h = x
    for _ in range(2):
        for _ in range(NL):
            h2 = one_layer(h).unsqueeze(0).expand(TOPK, Kgu).contiguous()
            h = h2
    torch.npu.synchronize()

    t0 = time.perf_counter()
    h = x
    for _ in range(NL):
        h2 = one_layer(h).unsqueeze(0).expand(TOPK, Kgu).contiguous()
        h = h2
    torch.npu.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    print(f"[real-chain] {NL} 层串联（每层 gate_up+act+down+加权）: {ms:.2f} ms")
    print(f"[real-chain] 每层 {ms/NL:.3f} ms；这是**单卡每 token 的专家段**（真形态，不可批量）")
    print(f"[real-chain] 对比：torch 路径 34.6 ms → {34.6/ms:.2f}x")


if __name__ == "__main__":
    main()
