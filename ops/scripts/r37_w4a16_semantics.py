import inspect
import time

import torch
import torch_npu

print("torch", torch.__version__, "| torch_npu", torch_npu.__version__)
f = torch_npu.npu_weight_quant_batchmatmul
doc = f.__doc__ or ""
keep = []
for ln in doc.splitlines():
    s = ln.strip()
    low = s.lower()
    if not s:
        continue
    if any(t in low for t in ("weight", "antiquant", "group", "int4", "int8", "int32", "重量")):
        keep.append(s[:190])
print("=== DOC 关键行 (%d) ===" % len(keep))
for s in keep[:32]:
    print("  |", s)
try:
    print("sig:", inspect.signature(f))
except Exception as e:  # noqa: BLE001
    print("sig?", e)

DEV = "npu:0"
torch.manual_seed(0)
K, N, E, gs = 2560, 1280, 10, 64
G = K // gs


def relerr(y, ref):
    y = y.float()
    ref = ref.float()
    d = (y - ref).abs()
    return float(d.max()), float(d.mean() / (ref.abs().mean() + 1e-9))


def verdict(rl):
    return "PASS" if rl < 2e-2 else "MISMATCH"


q0 = torch.randint(-8, 8, (E, N, K), dtype=torch.float32)
sgrp = q0.abs().view(E, N, G, gs).amax(-1) / 7.0 + 1e-9          # [E,N,G]
sexp = sgrp.unsqueeze(-1).expand(E, N, G, gs).reshape(E, N, K)
qi = torch.round(q0 / sexp).clamp_(-8, 7)
wdeq = (qi.view(E, N, G, gs) * sgrp.unsqueeze(-1)).reshape(E, N, K).to(torch.bfloat16)
x = torch.randn(E, K, dtype=torch.bfloat16, device=DEV)
ref = torch.bmm(x.float().unsqueeze(1), wdeq.float().transpose(1, 2)).squeeze(1)
SGN = sgrp.to(DEV)                     # [E,G,N]  per-group 文档布局
SGN16 = sgrp.half().to(DEV)

print("=== A) int8 weight [E,K,N] ===")
wi = qi.to(torch.int8).permute(0, 2, 1).contiguous().to(DEV)
for name, sc, kw in (
    ("scale[E,G,N] gs64 bf16", SGN, dict(antiquant_group_size=gs)),
    ("scale[E,G,N] gs64 fp16", SGN16, dict(antiquant_group_size=gs)),
    ("scale[E,N,G] gs64 bf16", sgrp.transpose(1, 2).contiguous().to(DEV), dict(antiquant_group_size=gs)),
    ("scale[E,N,1] gs0 perchan", sgrp[:, :, :1].contiguous().to(DEV), dict(antiquant_group_size=0)),
):
    try:
        y = f(x, wi, sc, **kw)
        mx, rl = relerr(y, ref)
        print("  %-26s max %.3e rel %.3e  %s" % (name, mx, rl, verdict(rl)))
    except Exception as e:  # noqa: BLE001
        print("  %-26s FAIL %s" % (name, str(e)[:150]))


def pack_nib(v, axis):
    u = (v & 0xF).to(torch.int64)
    if axis == "N":
        u = u.permute(0, 2, 1)          # [E,K,N] -> 沿 N 打包 -> [E,K,N/8]
    p = torch.zeros(u.shape[:-1] + (u.shape[-1] // 8,), dtype=torch.int64)
    for j in range(8):
        p += (u[..., j::8] & 0xF) << (4 * j)
    return p.to(torch.int32)


def pack_byte(v, axis):
    u = (v & 0xF).to(torch.int64)
    if axis == "N":
        u = u.permute(0, 2, 1)
    p = (u[..., 0::2] & 0xF) + ((u[..., 1::2] & 0xF) << 4)
    return p.to(torch.uint8)


w_n = pack_nib(qi.to(torch.int32), "N").to(DEV)     # [E,K,N/8] int32
w_k = pack_nib(qi.to(torch.int32), "K").to(DEV)     # [E,N,K/8] int32
w_b = pack_byte(qi.to(torch.int32), "N").to(DEV)    # [E,K,N/2] uint8
w_bk = pack_byte(qi.to(torch.int32), "K").to(DEV)   # [E,N,K/2] uint8
print("=== B) int4 各种容器 ===")
print("  shapes:", tuple(w_n.shape), tuple(w_k.shape), tuple(w_b.shape), tuple(w_bk.shape))
for name, wq, sc, kw in (
    ("int32 [K,N/8] scale[G,N]", w_n, SGN, dict(antiquant_group_size=gs)),
    ("int32 [K,N/8] scale[G,N] fp16", w_n, SGN16, dict(antiquant_group_size=gs)),
    ("int32 [N,K/8] scale[N,G]", w_k, sgrp.transpose(1, 2).contiguous().to(DEV), dict(antiquant_group_size=gs)),
    ("uint8 [K,N/2] scale[G,N]", w_b, SGN, dict(antiquant_group_size=gs)),
    ("uint8 [N,K/2] scale[N,G]", w_bk, sgrp.transpose(1, 2).contiguous().to(DEV), dict(antiquant_group_size=gs)),
    ("int32 [K,N/8] perchan gs0", w_n, sgrp[:, :, :1].contiguous().to(DEV), dict(antiquant_group_size=0)),
):
    try:
        y = f(x, wq, sc, **kw)
        mx, rl = relerr(y, ref)
        print("  %-30s max %.3e rel %.3e  %s" % (name, mx, rl, verdict(rl)))
    except Exception as e:  # noqa: BLE001
        print("  %-30s FAIL %s" % (name, str(e)[:150]))


def bench(fn, n=20):
    for _ in range(3):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


print("=== C) 时序 M=10 K=2560 N=1280（当前 torch 路径 MoE 约 1.0 ms/层）===")
for tag, wq, sc in (("int8 g64", wi, SGN), ("int4 int32 g64", w_n, SGN), ("int4 uint8 g64", w_b, SGN)):
    try:
        t = bench(lambda: f(x, wq, sc, antiquant_group_size=gs))
        print("  %-16s %.3f ms" % (tag, t))
    except Exception as e:  # noqa: BLE001
        print("  %-16s FAIL %s" % (tag, str(e)[:130]))
try:
    t = bench(lambda: f(x[:1], w_b[0], SGN[0], antiquant_group_size=gs))
    print("  int4 uint8 M=1  %.3f ms" % t)
except Exception as e:  # noqa: BLE001
    print("  int4 uint8 M=1 FAIL", str(e)[:140])
