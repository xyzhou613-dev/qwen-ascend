"""HBM 预算实测：每卡可用、单层实际占用构成。"""
import sys, torch
sys.path.insert(0, "/root/ascend-dev/fnx")
import torch_npu  # noqa
from fnx import engine as E
from fnx.load import load_layer, QDIR

for d in (0, 1):
    free, total = torch.npu.mem_get_info(d)
    print(f"npu:{d}  total {total/2**30:7.2f} GiB  free {free/2**30:7.2f} GiB  used {(total-free)/2**30:6.2f} GiB")

W = load_layer(f"{QDIR}/L00.safetensors", 0, "npu:0")
tot = 0
buckets = {}
for k, v in W.items():
    if isinstance(v, E.ExpQ):
        n = v.q.numel() + v.s.numel() * 2
        b = "experts(int8+fp16 scale)"
    elif isinstance(v, E.Raw):
        n = v.w.numel() * 2
        b = "dense(bf16)"
    else:
        n = v.numel() * v.element_size()
        b = "raw"
    buckets[b] = buckets.get(b, 0) + n
    tot += n
print(f"\nL00 权重合计 {tot/2**30:.3f} GiB")
for b, n in sorted(buckets.items(), key=lambda x: -x[1]):
    print(f"   {b:28s} {n/2**30:7.3f} GiB")
exp = buckets["experts(int8+fp16 scale)"]
print(f"\n外推 48 层: experts {exp*48/2**30:.1f} GiB, dense {buckets['dense(bf16)']*48/2**30:.1f} GiB")
print(f"  每卡(24层) ≈ {(exp+buckets['dense(bf16)'])*24/2**30:.1f} GiB  + embed/lm_head ≈ 1.8 GiB")
print(f"\n若 4bit 打包（2 值/字节）: experts 减半 → 每卡 ≈ {(exp/2+buckets['dense(bf16)'])*24/2**30:.1f} GiB")
print(f"若专家 scale 改 g128（scale 4→1）: experts {exp*0.969*48/2**30:.1f} GiB")
