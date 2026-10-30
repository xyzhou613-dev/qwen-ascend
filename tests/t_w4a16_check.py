"""t_w4a16_check: 新 exp 容器（aclnn per-group W4A16）与旧 torch 解包路径的等价性 + 时序。

用法（容器内，服务停掉）：
  FNX_MOE_W4A16=0 python tests/t_w4a16_check.py
会分别以 flag=0/1 各加载一次 L00，然后：
  1) 同一组专家 ids、同一 x，比较 gu / dn 的输出（期望 bf16 舍入级一致）
  2) 比较两条路径的"一层 MoE"耗时
"""
import os
import time

# ★ 必须与生产同口径：g64 + LQP artifact，否则对比的是 g32 的原始 Q 权重，数值根本不可比
os.environ.setdefault("FNX_PACKED_TORCH", "1")
os.environ.setdefault("FNX_EXP_G", "64")
os.environ.setdefault("FNX_LQP", "/models/Qwen3.8-Flash-Next-LQP")

import torch

LAYER = os.environ.get("FNX_QDIR", "/models/Qwen3.8-Flash-Next-Q") + "/L00.safetensors"
DEV = "npu:0"


def tm(fn, n=20, warm=4):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def relerr(y, ref):
    y = y.float()
    ref = ref.float()
    d = (y - ref).abs()
    return float(d.max()), float(d.mean() / (ref.abs().mean() + 1e-9))


def main():
    import torch_npu  # noqa: F401
    from fnx import engine as E
    from fnx.load import load_layer

    os.environ["FNX_MOE_W4A16"] = "0"
    old = load_layer(LAYER, 0, DEV)
    os.environ["FNX_MOE_W4A16"] = "1"
    new = load_layer(LAYER, 0, DEV)
    gu_o, gu_n = old["mlp.experts.gate_up_proj"], new["mlp.experts.gate_up_proj"]
    dn_o, dn_n = old["mlp.experts.down_proj"], new["mlp.experts.down_proj"]
    print("旧容器 %s  新容器 %s" % (type(gu_o).__name__, type(gu_n).__name__))
    assert isinstance(gu_n, E.ExpW4)

    H = E.H
    torch.manual_seed(0)
    ids = torch.tensor([3, 17, 61, 200, 255, 256, 400, 401, 510, 511],
                       dtype=torch.int64, device=DEV)
    x = torch.randn(1, H, dtype=torch.bfloat16, device=DEV)

    y_old = gu_o.matmul(x.to(torch.bfloat16).expand(E.TOPK, 1, H), ids)      # [10,N]
    y_new = gu_n.apply(x, ids).view(E.TOPK, gu_n.N)
    mx, rl = relerr(y_new, y_old)
    print("gu  max %.3e rel %.3e  %s" % (mx, rl, "PASS" if rl < 3e-2 else "FAIL"))

    act = torch.randn(E.TOPK, E.MI, dtype=torch.bfloat16, device=DEV) * 0.1
    d_old = dn_o.matmul(act.unsqueeze(1), ids)                              # [10,N2]
    d_new = dn_n.apply_diag(act, ids)
    mx2, rl2 = relerr(d_new, d_old)
    print("dn  max %.3e rel %.3e  %s" % (mx2, rl2, "PASS" if rl2 < 3e-2 else "FAIL"))

    t_old = tm(lambda: (gu_o.matmul(x.expand(E.TOPK, 1, H), ids),
                        dn_o.matmul(act.unsqueeze(1), ids)))
    t_new = tm(lambda: (gu_n.apply(x, ids), dn_n.apply_diag(act, ids)))
    print("一层专家 GEMM：旧 %.3f ms  新 %.3f ms  (%.2fx)" % (t_old, t_new, t_old / t_new))


if __name__ == "__main__":
    with torch.no_grad():
        main()
