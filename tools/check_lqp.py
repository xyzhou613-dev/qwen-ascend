"""★ 校验 LQP artifact 与"运行时打包"是否等价（数值层面，因为两种编码字节不同）。

LQP 用 offset-binary（n'=value+8），运行时用二进制补码（n=value&0xF）。
两者解码后必须给出**完全相同**的 (wlo, whi)。

用法：cd /root/ascend-dev/fnx && python tools/check_lqp.py
"""
import os

import numpy as np
import torch
import torch_npu  # noqa: F401
from safetensors import safe_open
from safetensors.torch import load_file

from fnx import engine as E
from fnx.load import QDIR
from fnx.w4gemv_op import W4GemvOp

LQP = os.environ.get("FNX_LQP", "/models/Qwen3.8-Flash-Next-LQP")
DEV = "npu:0"
LAYER = 0
PROJ = "gate_up_proj"


def main():
    print(f"=== LQP vs 运行时打包（L{LAYER:02d} {PROJ}）===", flush=True)
    # 1) artifact 侧
    a = load_file(os.path.join(LQP, f"L{LAYER:02d}.safetensors"))
    k = f"mlp.experts.{PROJ}"
    eq_lqp = E.ExpQ.from_packed(a[k + ".v2o"], a[k + ".s8"], a[k + ".base"], DEV)
    print(f"  LQP: E={eq_lqp.qv2.shape[0]} N={eq_lqp.N} K={eq_lqp.K} "
          f"v2ob={eq_lqp.v2ob}", flush=True)

    # 2) 运行时侧：从 QP 重建同样的 q2/s8/base，再走 to_v2_layout（二进制补码）
    with safe_open(os.path.join(QDIR, f"L{LAYER:02d}.safetensors"),
                   framework="pt", device="cpu") as f:
        keys = list(f.keys())
        kq = [x for x in keys if x.endswith(f"{PROJ}.q4")][0]
        ks = [x for x in keys if x.endswith(f"{PROJ}.s4")][0]
        q4, s4 = f.get_tensor(kq), f.get_tensor(ks)
    q2, s8r, baser = E.merge_scales_g64(q4, s4, gs_in=q4.shape[-1] // s4.shape[-1], gs_out=64)
    p_rt = torch.from_numpy(np.ascontiguousarray(
        W4GemvOp.to_v2_layout(q2.numpy(), q2.shape[1], q2.shape[2]))).to(DEV)
    eq_rt = E.ExpQ.from_packed(p_rt, s8r, baser, DEV)
    eq_rt.v2ob = False                      # 运行时打包是二进制补码

    # 3) 尺度一致性
    ds = (eq_lqp.s.float() - eq_rt.s.float()).abs()
    db = (eq_lqp.base.float() - eq_rt.base.float()).abs()
    print(f"  s8   差异 max {float(ds.max()):.3e}", flush=True)
    print(f"  base 差异 max {float(db.max()):.3e}", flush=True)
    ok = True
    if float(ds.max()) > 0 or float(db.max()) > 0:
        print("  !! 尺度不一致（artifact 与运行时 merge 结果不同）", flush=True)
        ok = False

    # 4) 解码出的权重必须逐元素相同
    idx = torch.tensor([0, 1, 7, 63, 100, 255, 511], dtype=torch.long, device=DEV)
    idx = idx[idx < eq_lqp.qv2.shape[0]]
    wlo1, whi1 = eq_lqp._deq_v2(idx)
    wlo2, whi2 = eq_rt._deq_v2(idx)
    for nm, x, y in (("wlo", wlo1, wlo2), ("whi", whi1, whi2)):
        d = (x.float() - y.float()).abs()
        rel = float(d.mean() / x.float().abs().mean().clamp_min(1e-12))
        print(f"  {nm}: rel {rel*100:.6f}%  max {float(d.max()):.3e}", flush=True)
        if rel > 1e-6:
            ok = False
    print(f"\n  判定：{'完全等价 ✅' if ok else '不等价 ❌'}", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
