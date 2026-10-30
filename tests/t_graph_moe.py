"""MoE 专家段的 NPUGraph 捕获 PoC：24 层 chain 整体入图，对比 eager vs replay。

照抄现有服务 model.py:142-163 的捕获写法（side stream warmup + relaxed）。
目的：验证"每层 ~0.21 ms 的 Python 胶水"能否被图吃掉。
"""
import json
import time

import numpy as np
import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401

from fnx.w4gemv_op import W4GemvOp, GS

GOLD = "/root/ascend-dev/golden/w4gemv"
NL, TOPK = 24, 10


def load(name):
    meta = json.load(open(GOLD + "/meta.json"))
    c = meta["cases"]["multi_" + name + "_proj"]
    nE, N, K = c["B"], c["N"], c["K"]
    p4 = np.fromfile(f"{GOLD}/multi/{name}_proj_p4.bin", dtype=np.int8).reshape(nE, N, K // 2)
    s4 = np.fromfile(f"{GOLD}/multi/{name}_proj_s4.bin", dtype=np.float16).reshape(nE, N, K // GS)
    return p4, s4, nE, N, K


def main():
    op = W4GemvOp()
    gu_p4, gu_s4, nE, Ngu, Kgu = load("gate_up")
    dn_p4, dn_s4, _, Ndn, Kdn = load("down")
    gu_p4 = torch.from_numpy(np.ascontiguousarray(op.to_v2_layout(gu_p4, Ngu, Kgu))).npu()
    dn_p4 = torch.from_numpy(np.ascontiguousarray(op.to_v2_layout(dn_p4, Ndn, Kdn))).npu()
    gu_s4t, dn_s4t = op.prepare_layer(gu_s4), op.prepare_layer(dn_s4)
    del gu_s4, dn_s4

    # 静态 buffer（图捕获要求地址不变）
    x_buf = torch.randn(TOPK, Kgu, dtype=torch.float32, device="npu:0")
    out_gu = torch.empty((TOPK, Ngu), dtype=torch.float32, device="npu:0")
    out_dn = torch.empty((TOPK, Ndn), dtype=torch.float32, device="npu:0")
    wdev = torch.rand(TOPK, device="npu:0")

    def one_layer():
        y = op.run(gu_p4, gu_s4t, x_buf, nE, Ngu, Kgu, lay=2, out=out_gu)
        act = (F.silu(y[:, :Ngu // 2]) * y[:, Ngu // 2:]).contiguous()
        y2 = op.run(dn_p4, dn_s4t, act, nE, Ndn, Kdn, lay=2, out=out_dn)
        s = (y2 * wdev.unsqueeze(-1)).sum(0)
        x_buf.copy_(s.unsqueeze(0).expand(TOPK, Kgu))

    def chain():
        for _ in range(NL):
            one_layer()

    # eager 基线
    with torch.no_grad():
        chain(); torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            chain()
        torch.npu.synchronize()
        eager = (time.perf_counter() - t0) / 5 * 1e3
        print(f"[graph] eager 24 层: {eager:.2f} ms  ({eager/NL:.3f} ms/层)")

        # 捕获（照 model.py 的写法）
        s = torch.npu.Stream()
        try:
            with torch.npu.stream(s):
                for _ in range(3):
                    chain()
                torch.npu.synchronize()
                g = torch.npu.NPUGraph()
                with torch.npu.graph(g, capture_error_mode="relaxed"):
                    chain()
                torch.npu.synchronize()
            torch.npu.synchronize()
            print("[graph] 捕获成功")
        except Exception as e:  # noqa: BLE001
            print(f"[graph] 捕获失败: {type(e).__name__}: {e}")
            return

        # replay 计时
        for _ in range(2):
            g.replay()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(10):
            g.replay()
        torch.npu.synchronize()
        rep = (time.perf_counter() - t0) / 10 * 1e3
        print(f"[graph] replay 24 层: {rep:.2f} ms  ({rep/NL:.3f} ms/层)")
        print(f"[graph] === 提速 {100*(eager-rep)/eager:.1f}%  "
              f"(eager {eager:.2f} -> replay {rep:.2f} ms) ===")


if __name__ == "__main__":
    main()
