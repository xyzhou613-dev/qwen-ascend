"""验证 507015 的根因假设：W4GemvOp 的 _dbg 固定在 npu:0，而计算在 npu:1 → 跨卡指针。

三个组合：
  A) 全部在 npu:0            → 应 PASS
  B) 数据在 npu:1，_dbg 在 npu:0（当前实现） → 预期 507015
  C) 数据在 npu:1，_dbg 也在 npu:1          → 应 PASS
"""
import json

import numpy as np
import torch
import torch_npu  # noqa: F401

from fnx.w4gemv_op import W4GemvOp, GS

GOLD = "/root/ascend-dev/golden/w4gemv"


def data():
    meta = json.load(open(GOLD + "/meta.json"))
    c = meta["cases"]["multi_gate_up_proj"]
    nE, N, K = c["B"], c["N"], c["K"]
    p4 = np.fromfile(f"{GOLD}/multi/gate_up_proj_p4.bin", dtype=np.int8).reshape(nE, N, K // 2)
    s4 = np.fromfile(f"{GOLD}/multi/gate_up_proj_s4.bin", dtype=np.float16).reshape(nE, N, K // GS)
    x = ((np.fromfile(f"{GOLD}/multi/gate_up_proj_x.bin", dtype=np.int16).astype(np.uint32) << 16)
         .view(np.float32)).reshape(nE, K)
    return p4, s4, x, nE, N, K


def run_case(tag, dev, dbg_dev):
    p4, s4, x, nE, N, K = data()
    op = W4GemvOp()
    if dbg_dev != 0:
        op._dbg = torch.zeros(65536, dtype=torch.float32, device=f"npu:{dbg_dev}")   # 换到目标卡
    s4t = op.prepare_layer(s4.astype(np.float16))
    if dev != 0:
        s4t = s4t.to(f"npu:{dev}")
    p4v2 = torch.from_numpy(np.ascontiguousarray(op.to_v2_layout(p4, N, K))).to(f"npu:{dev}")
    xd = torch.from_numpy(np.ascontiguousarray(x)).to(f"npu:{dev}")
    try:
        y = op.run(p4v2, s4t, xd, nE, N, K, lay=2).cpu().numpy()
        print(f"[{tag}] dev={dev} dbg_dev={dbg_dev} -> OK  sum={y.sum():.3f}", flush=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[{tag}] dev={dev} dbg_dev={dbg_dev} -> 异常 {type(e).__name__}: {str(e)[:160]}", flush=True)
        return False


if __name__ == "__main__":
    run_case("A 同卡0", 0, 0)
    torch.npu.synchronize()
    okb = run_case("B 数据1/dbg0", 1, 0)     # 预期失败
    torch.npu.synchronize()
    okc = run_case("C 数据1/dbg1", 1, 1)     # 预期成功
    torch.npu.synchronize()
    print(f"=== 结论：B={'挂' if not okb else '没挂'}  C={'通' if okc else '挂了'} "
          f"→ 跨卡假设{'成立' if (not okb and okc) else '不成立'} ===")
