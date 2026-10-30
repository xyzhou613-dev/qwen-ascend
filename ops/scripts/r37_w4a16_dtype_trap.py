import traceback

import torch
import torch_npu

DEV = "npu:0"
torch.manual_seed(0)
f = torch_npu.npu_weight_quant_batchmatmul
doc = f.__doc__ or ""

i = doc.find("调用示例")
print("=== doc 示例段（前 2600 字符）===")
print(doc[i:i + 2600] if i >= 0 else doc[-2600:])

K, N, gs = 2560, 1280, 64
G = K // gs
W = torch.randn(N, K) * 0.02
s = (W.abs().view(N, G, gs).amax(-1) / 7.0 + 1e-9).float()
q = torch.round(W / s.unsqueeze(-1).repeat(1, 1, gs).reshape(N, K)).clamp_(-8, 7)
wdeq = (q.view(N, G, gs) * s.unsqueeze(-1)).reshape(N, K).to(torch.bfloat16)
x16 = torch.randn(1, K, dtype=torch.bfloat16, device=DEV)
x32 = x16.float()
ref = (x16.float().cpu() @ wdeq.float().t())
sc_nd = s.t().contiguous().to(DEV).to(torch.bfloat16)      # [G,N]
w8 = q.to(torch.int8).t().contiguous().to(DEV)             # [K,N]
w32 = q.to(torch.int32).t().contiguous().to(DEV)           # [K,N]


def run(tag, **kw):
    try:
        y = f(**kw)
        d = (y.float().cpu() - ref).abs()
        rl = float(d.mean() / (ref.abs().mean() + 1e-9))
        print("  %-42s OK shape %-16s rel %.3e %s"
              % (tag, tuple(y.shape), rl, "PASS" if rl < 2e-2 else "MISMATCH"))
        return y
    except Exception:  # noqa: BLE001
        tb = traceback.format_exc().splitlines()
        msg = " | ".join(ln.strip() for ln in tb[-9:] if ln.strip())
        print("  %-42s FAIL %s" % (tag, msg[-700:]))
        return None


print("\n=== int8 per-group ===")
run("int8 ND scale[G,N] gs64", x=x16, weight=w8, antiquant_scale=sc_nd, antiquant_group_size=gs)
run("int8 ND scale[G,N] gs64 inner=1", x=x16, weight=w8, antiquant_scale=sc_nd, antiquant_group_size=gs, inner_precise=1)
run("int8 ND scale fp16", x=x16, weight=w8, antiquant_scale=sc_nd.half(), antiquant_group_size=gs)
try:
    w8nz = torch_npu.npu_format_cast(w8, 29)
    print("  int8 NZ ok, format:", torch_npu.get_npu_format(w8nz))
    run("int8 NZ scale[G,N] gs64", x=x16, weight=w8nz, antiquant_scale=sc_nd, antiquant_group_size=gs)
except Exception:  # noqa: BLE001
    print("  int8 NZ cast FAIL", traceback.format_exc().splitlines()[-1][:200])

print("\n=== int4 int32 (int4pack) per-group ===")
wpack = torch_npu.npu_convert_weight_to_int4pack(w32)
print("  int4pack [K,N] ->", tuple(wpack.shape), wpack.dtype)
run("int4 ND scale[G,N] gs64", x=x16, weight=wpack, antiquant_scale=sc_nd, antiquant_group_size=gs)
run("int4 ND scale[G,N] gs64 inner=1", x=x16, weight=wpack, antiquant_scale=sc_nd, antiquant_group_size=gs, inner_precise=1)
run("int4 ND scale fp16", x=x16, weight=wpack, antiquant_scale=sc_nd.half(), antiquant_group_size=gs)
try:
    wpnz = torch_npu.npu_format_cast(wpack, 29)
    print("  int4 NZ ok, format:", torch_npu.get_npu_format(wpnz))
    run("int4 NZ scale[G,N] gs64", x=x16, weight=wpnz, antiquant_scale=sc_nd, antiquant_group_size=gs)
    run("int4 NZ scale[G,N] gs64 inner=1", x=x16, weight=wpnz, antiquant_scale=sc_nd,
        antiquant_group_size=gs, inner_precise=1)
except Exception:  # noqa: BLE001
    print("  int4 NZ cast FAIL", traceback.format_exc().splitlines()[-1][:200])

print("\n=== x 转置形态（doc：NZ per-group 要求 x 转置输入）===")
xt = x16.t().contiguous()                                   # [K,1]
run("int4 ND x=[K,1]", x=xt, weight=wpack, antiquant_scale=sc_nd, antiquant_group_size=gs)
