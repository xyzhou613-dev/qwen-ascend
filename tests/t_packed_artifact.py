"""校验落盘的真 4-bit 打包专家权重：文件 → ExpQ(packed) 反量化 == 原 int8 路径反量化。"""
import sys, torch
sys.path.insert(0, "/root/ascend-dev/fnx")
import torch_npu  # noqa
from fnx import engine as E
from fnx.load import load_layer, load_packed_experts, QDIR, QPACKED, SRCDIR
import os

L = int(sys.argv[1]) if len(sys.argv) > 1 else 0
DEV = "npu:0"
W = load_layer(os.path.join(QDIR, f"L{L:02d}.safetensors"), L, DEV)
pk = load_packed_experts(L, DEV)

idx = torch.tensor([7, 343, 105, 212, 243, 254, 403, 168, 302, 67], device=DEV)
for which, key in (("gate_up_proj", "mlp.experts.gate_up_proj"), ("down_proj", "mlp.experts.down_proj")):
    ref = W[key]
    p4, s4 = pk[which]
    eq = E.ExpQ.__new__(E.ExpQ)
    eq.q, eq.s, eq.gs, eq.K, eq.packed = p4, s4, ref.q.shape[-1] // ref.s.shape[-1], ref.q.shape[-1], True
    a = ref.gather_deq(idx)          # 未打包 int8 路径
    b = eq.gather_deq(idx)           # 落盘打包文件路径
    d = (a.float() - b.float()).abs()
    print(f"  {which:14s} p4 {tuple(p4.shape)} s4 {tuple(s4.shape)}  "
          f"torch.equal={torch.equal(a, b)}  maxdiff={d.max().item():.3e}", flush=True)
print("manifest:", os.path.exists(os.path.join(QPACKED, "manifest.json")))
print("OK")