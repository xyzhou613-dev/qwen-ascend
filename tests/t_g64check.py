"""单层验证 merge_scales_g64：数值增量误差 + 耗时（先小尺度验证，再上整机）。"""
import sys, time
import torch
sys.path.insert(0, "/root/ascend-dev/fnx")
import torch_npu  # noqa
from fnx import engine as E
from fnx.load import load_layer, QDIR

DEV = "npu:0"
W = load_layer(f"{QDIR}/L00.safetensors", 0, DEV)
idx = torch.tensor([7, 343, 105, 212, 243, 254, 403, 168, 302, 67], device=DEV)
torch.manual_seed(0)

for key in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
    ref = W[key]                                   # g32 fp16 原路径
    q, s, gs = ref.q, ref.s, ref.gs
    a = ref.gather_deq(idx)                        # 参考：g32 反量化
    torch.npu.synchronize()
    base_mem = torch.npu.memory_allocated(0)
    t = time.time()
    q2, s8, base = E.merge_scales_g64(q.clone(), s, gs_in=gs, gs_out=64)
    torch.npu.synchronize()
    dt = time.time() - t
    eq = E.ExpQ(q2, s8, 64, packed=False)
    eq.base = base
    b = eq.gather_deq(idx)
    torch.npu.synchronize()
    peak = torch.npu.memory_allocated(0) - base_mem
    d = (a.float() - b.float()).abs()
    rel = d.mean().item() / a.float().abs().mean().clamp_min(1e-9).item()
    print(f"{key.split('.')[-1]:14s} q {tuple(q.shape)} gs {gs}->64  "
          f"增量 rel {rel*100:6.3f}%  max {d.max().item():.3e}  "
          f"耗时 {dt*1e3:6.1f} ms  常驻增量 {peak/2**20:7.1f} MiB", flush=True)

print("\n每层专家 scale 字节：g32 fp16 ->", flush=True)
for key in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
    ref = W[key]
    n32 = ref.s.numel() * 2
    print(f"  {key.split('.')[-1]:14s} {n32/2**20:7.2f} MiB -> 预计 {n32/4/2**20:6.2f} MiB(+基准)", flush=True)
print("OK")
