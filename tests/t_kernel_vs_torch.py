"""kernel 路径 vs 线上 torch 路径（dequant + bmm，bf16）在真实 L00 权重上对拍。

为什么要单独做：之前的参考都是 numpy fp32 契约，抓不到 bf16 舍入差。
而线上跑的是 bf16（`ExpQ.gather_deq` 返回 bf16，bmm 也是 bf16），
所以必须比一次 bf16 口径，确认"换成 kernel 不会引入额外误差"。
"""
import os

import numpy as np
import torch
import torch_npu  # noqa: F401
from safetensors import safe_open

from fnx.w4gemv_op import W4GemvOp, GS

QP = "/models/Qwen3.8-Flash-Next-QP"
LAYER = "L00"
nE = 10


def torch_path(p4_u8, s4, x_bf16):
    """复刻 engine.py 的 ExpQ.gather_deq + bmm（bf16 口径）。"""
    k, N = p4_u8.shape[0], p4_u8.shape[1]
    K = p4_u8.shape[2] * 2
    p4t = torch.from_numpy(np.ascontiguousarray(p4_u8)).npu() if isinstance(p4_u8, np.ndarray) else p4_u8.to("npu:0")
    s4_t = torch.from_numpy(np.ascontiguousarray(s4)).npu() if isinstance(s4, np.ndarray) else s4.to("npu:0")
    p = p4t.to(torch.int16) & 0xFF
    lo = ((p & 0x0F) ^ 8) - 8
    hi = (((p >> 4) & 0x0F) ^ 8) - 8
    q = torch.stack([lo, hi], dim=-1).reshape(k, N, K).to(torch.bfloat16)
    s = s4_t
    w = q.view(k, N, K // GS, GS) * s.to(torch.bfloat16).unsqueeze(-1)
    w = w.reshape(k, N, K)
    xb = x_bf16.to("npu:0").expand(k, K).unsqueeze(1)          # [k,1,K]
    return torch.bmm(xb, w.transpose(1, 2)).squeeze(1).float()  # [k,N] fp32
    # 注：K 维必须按 k 连续（fp32 参考里用 einsum 已验过），这里复刻线上口径


def main():
    op = W4GemvOp()
    path = os.path.join(QP, LAYER + ".packed.safetensors")
    rng = np.random.RandomState(0)
    with safe_open(path, framework="pt") as f:
        for want in ("gate_up_proj", "down_proj"):
            kp = [k for k in f.keys() if want in k and k.endswith(".p4")][0]
            ks = [k for k in f.keys() if want in k and k.endswith(".s4")][0]
            p4 = f.get_slice(kp)[0:nE].numpy().reshape(nE, -1, f.get_slice(kp).get_shape()[2])
            s4 = f.get_slice(ks)[0:nE].numpy()
            N, Kh = p4.shape[1], p4.shape[2] * 2
            x = (rng.randn(Kh).astype(np.float32) * 0.3)

            # 线上 torch 路径（bf16）
            y_t = torch_path(p4, s4, torch.from_numpy(x))
            # kernel 路径（v2 布局）
            p4v2 = torch.from_numpy(np.ascontiguousarray(op.to_v2_layout(p4, N, Kh))).npu()
            s4t = op.prepare_layer(s4.astype(np.float16))
            xd = torch.from_numpy(np.ascontiguousarray(np.repeat(x[None, :], nE, 0))).npu()
            y_k = op.run(p4v2, s4t, xd, nE, N, Kh, lay=2).float()

            d = (y_k - y_t).abs()
            scale = y_t.abs().mean() + 1e-12
            rel = float(d.mean() / scale)
            print(f"{want:14s} N={N} K={Kh}  kernel vs torch(bf16): "
                  f"rel_mean {rel*100:.3f}%  max {float(d.max()):.4f}  "
                  f"-> {'PASS(bf16 噪声级)' if rel < 5e-3 else 'CHECK'}", flush=True)
            del p4v2, s4t, xd, y_k, y_t


if __name__ == "__main__":
    main()
