"""★ 门禁：真实 QP 权重 + g64 压缩 scale（`s` 是 int8，真实 scale = s*base）下，
kernel 路径必须和线上 torch 路径（`ExpQ.gather_deq`）一致。

为什么必须有这条：`t_kernel_vs_torch.py` 用的是 packed safetensors 里的 **fp16** s4，
根本没有 base，所以它对"kernel 漏乘 base"这个 bug 完全免疫 —— 实测漏乘后线上 logits 全 NaN，
而那条测试仍然 PASS。本测试走 `load_layer` 的同一条真实链路（q4/s4 -> merge_scales_g64 -> ExpQ(base)）。
"""
import os

import numpy as np
import torch
import torch_npu  # noqa: F401
from safetensors.torch import load_file

from fnx import engine as E
from fnx.load import QDIR
from fnx.w4gemv_op import W4GemvOp

LAYER = 0
nE = 10


def main():
    op = W4GemvOp()
    sd = load_file(os.path.join(QDIR, f"L{LAYER:02d}.safetensors"))
    rng = np.random.RandomState(0)
    ok_all = True
    expect = {"gate_up_proj": (1280, 2560), "down_proj": (2560, 640)}
    for want, (Nexp, Kexp) in expect.items():
        kq = [k for k in sd if want in k and k.endswith(".q4")][0]
        ks = [k for k in sd if want in k and k.endswith(".s4")][0]
        q4 = sd[kq][0:nE]
        s4 = sd[ks][0:nE]
        assert q4.dim() == 3, q4.shape
        E_tot, N, K = q4.shape
        assert (N, K) == (Nexp, Kexp), f"{want} 形状 {q4.shape} 与预期 {(Nexp, Kexp)} 不符"
        gs_in = K // s4.shape[-1]

        # 与 load.py 完全一致的合并：fp16 g32 scale -> int8 g64 + 每行 fp16 base
        q2, s8, base = E.merge_scales_g64(q4, s4, gs_in=gs_in, gs_out=64)
        q2d, s8d, based = q2.to("npu:0"), s8.to("npu:0"), base.to("npu:0")

        # ---- 线上 torch 路径 ----
        os.environ["FNX_MOE_KERNEL"] = "0"
        eq_t = E.ExpQ(q2d, s8d, 64, packed=False, base=based)
        idx = torch.arange(nE, dtype=torch.int64, device="npu:0")
        w = eq_t.gather_deq(idx)                                   # [nE,N,K] bf16
        x = (rng.randn(K).astype(np.float32) * 0.3)
        xb = torch.from_numpy(np.ascontiguousarray(np.repeat(x[None, :], nE, 0))).npu()
        y_t = torch.bmm(xb.to(torch.bfloat16).unsqueeze(1), w.transpose(1, 2)).squeeze(1).float()

        # ---- kernel 路径（同一份 q2/s8/base）----
        os.environ["FNX_MOE_KERNEL"] = "1"
        eq_k = E.ExpQ(q2d, s8d, 64, packed=False, base=based)
        assert eq_k.s4t is not None and eq_k.qv2 is not None, "kernel 权重没建出来"
        y_k = op.run(eq_k.qv2, eq_k.s4t, xb, nE, N, K, lay=2).float()

        d = (y_k - y_t).abs()
        rel = float(d.mean() / (y_t.abs().mean() + 1e-12))
        ok = (rel < 5e-3) and not bool(torch.isnan(y_k).any())
        ok_all &= ok
        print(f"{want:14s} E={E_tot} N={N} K={K} gs_in={gs_in}  "
              f"kernel vs torch(bf16) rel_mean {rel*100:.4f}%  max {float(d.max()):.4f}  "
              f"nan_k={int(torch.isnan(y_k).sum())} -> {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            print(f"  !! y_k|mean|={float(y_k.abs().mean()):.4e} y_t|mean|={float(y_t.abs().mean()):.4e}",
                  flush=True)
        del w, y_t, y_k, eq_t, eq_k, q2d, s8d, based
    print("RESULT:", "PASS" if ok_all else "FAIL", flush=True)


if __name__ == "__main__":
    main()
