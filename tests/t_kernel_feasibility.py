"""★ kernel 路径的两个前提核实（第 33 轮）：

A) kernel vs torch 在真实 L00 权重上的单层差异（复跑 tests/t_kernel_vs_torch.py 的核心）
B) FNX_MOE_KERNEL=1 与 LQP artifact（from_packed）是否兼容？
   代码事实：`from_packed` 只设 qv2，**从不构造 s4t**；而 `_moe` 的 kernel 分支要 `self.exp_gu.s4t`。
   ⇒ 预期：kernel + LQP 会拿到 s4t=None 而报错/走空。
C) kernel 需要 g32 fp16 的 s4t（`prepare_layer`），而 artifact 里是 g128 int8 + base。
   算一下把 s4t 造出来要多少显存。
"""
import numpy as np
import torch
import torch_npu  # noqa: F401
from safetensors import safe_open

from fnx import engine as E
from fnx.w4gemv_op import W4GemvOp, GS

QP = "/models/Qwen3.8-Flash-Next-QP"


def torch_path(p4_u8, s4, x_bf16):
    k, N = p4_u8.shape[0], p4_u8.shape[1]
    K = p4_u8.shape[2] * 2
    p4t = torch.from_numpy(np.ascontiguousarray(p4_u8)).npu()
    s4_t = torch.from_numpy(np.ascontiguousarray(s4)).npu()
    p = p4t.to(torch.int16) & 0xFF
    lo = ((p & 0x0F) ^ 8) - 8
    hi = (((p >> 4) & 0x0F) ^ 8) - 8
    q = torch.stack([lo, hi], dim=-1).reshape(k, N, K).to(torch.bfloat16)
    w = (q.view(k, N, K // GS, GS) * s4_t.to(torch.bfloat16).unsqueeze(-1)).reshape(k, N, K)
    xb = x_bf16.to("npu:0").expand(k, K).unsqueeze(1)
    return torch.bmm(xb, w.transpose(1, 2)).squeeze(1).float()


def main():
    print("=== A) kernel vs torch（真实 L00 权重，bf16 口径）===", flush=True)
    op = W4GemvOp()
    path = f"{QP}/L00.packed.safetensors"
    rng = np.random.RandomState(0)
    with safe_open(path, framework="pt") as f:
        for want in ("gate_up_proj", "down_proj"):
            kp = [k for k in f.keys() if want in k and k.endswith(".p4")][0]
            ks = [k for k in f.keys() if want in k and k.endswith(".s4")][0]
            sl = f.get_slice(kp)
            p4 = sl[0:10].numpy().reshape(10, -1, sl.get_shape()[2])
            s4 = f.get_slice(ks)[0:10].numpy()
            N, Kh = p4.shape[1], p4.shape[2] * 2
            x = (rng.randn(Kh).astype(np.float32) * 0.3)
            y_t = torch_path(p4, s4, torch.from_numpy(x))
            p4v2 = torch.from_numpy(np.ascontiguousarray(op.to_v2_layout(p4, N, Kh))).npu()
            s4t = op.prepare_layer(s4.astype(np.float16))
            xd = torch.from_numpy(np.ascontiguousarray(np.repeat(x[None, :], 10, 0))).npu()
            y_k = op.run(p4v2, s4t, xd, 10, N, Kh, lay=2).float()
            d = (y_k - y_t).abs()
            rel = float(d.mean() / (y_t.abs().mean() + 1e-12))
            print(f"  {want:14s} N={N} K={Kh}  rel_mean {rel*100:.3f}%  "
                  f"max|Δ| {float(d.max()):.4f}", flush=True)
            del p4v2, s4t, xd, y_k, y_t

    print("\n=== B) kernel + LQP artifact 兼容性 ===", flush=True)
    lqp = "/models/Qwen3.8-Flash-Next-LQP-g128/L00.safetensors"
    with safe_open(lqp, framework="pt") as f:
        qv2 = f.get_tensor("mlp.experts.gate_up_proj.v2o")
        s8 = f.get_tensor("mlp.experts.gate_up_proj.s8")
        base = f.get_tensor("mlp.experts.gate_up_proj.base")
    eq = E.ExpQ.from_packed(qv2, s8, base, "npu:0")
    print(f"  from_packed: qv2={tuple(eq.qv2.shape)} s={tuple(eq.s.shape)} gs={eq.gs}")
    print(f"  s4t = {eq.s4t}   ← kernel 分支要的就是这个")
    miss = [a for a in ("s4t",) if getattr(eq, a, None) is None]
    print(f"  ⇒ kernel 路径缺字段: {miss}  "
          f"（结论：{'不兼容，必须先把 s8/base 展开成 g32 fp16 的 s4t' if miss else '可用'}）")

    print("\n=== C) 造 s4t 的显存代价 ===", flush=True)
    E_, N, Kh = 512, 1280, 2560
    per_layer = E_ * N * (Kh // 32) * 2 / 2**20
    print(f"  gate_up: {E_} 专家 × {N} 行 × {Kh//32} 组 × 2 B = {per_layer:.0f} MiB/层")
    dn = E_ * 2560 * (640 // 32) * 2 / 2**20
    print(f"  down   : {dn:.0f} MiB/层")
    print(f"  合计 ≈ {per_layer+dn:.0f} MiB/层，45 层 = {(per_layer+dn)*45/1024:.2f} GiB")
    print(f"  （当前 s8+base 只占 22.6 MiB/层）")
    print(f"  ⇒ 展开成 s4t 每层多 {(per_layer+dn)-22.6:.0f} MiB，45 层多 {(per_layer+dn-22.6)*45/1024:.2f} GiB")


if __name__ == "__main__":
    with torch.no_grad():
        main()
