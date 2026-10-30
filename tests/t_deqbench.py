"""反量化吞吐微基准：找出 59 GB/s 慢在哪，选最快的反量化写法。"""
import sys, time
import torch
sys.path.insert(0, "/root/ascend-dev/fnx")
import torch_npu  # noqa
from fnx import engine as E

DEV = "npu:0"
E_N, N, K = 10, 1280, 2560
gs = 32

torch.manual_seed(0)
q = torch.randint(-8, 8, (E_N, N, K), dtype=torch.int8, device=DEV)
s = (torch.rand(E_N, N, K // gs, dtype=torch.float32, device=DEV) * 0.01).to(torch.float16)
x = torch.randn(E_N, 1, K, dtype=torch.bfloat16, device=DEV) / 20


def bench(fn, n=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


nbytes_in = q.numel() * 1
nbytes_out = q.numel() * 2
traffic = nbytes_in + nbytes_out      # read int8 + write bf16 (ideal, 1 pass)


def rep(name, fn, extra_out=0):
    t = bench(fn)
    gb = (traffic + extra_out) / 1e9
    print(f"  {name:52s} {t:7.3f} ms   {gb/(t/1e3):7.1f} GB/s(理想口径)  {q.numel()/1e9/(t/1e3):6.1f} Gparam/s")
    return t


print(f"专家 gate_up 形状 {tuple(q.shape)}  gs={gs}  参数量 {q.numel()/1e6:.1f}M")
print("\n--- 1) 纯转换，不乘 scale ---")
rep("q.to(bf16)                       ", lambda: q.to(torch.bfloat16))
rep("q.to(fp16)                       ", lambda: q.to(torch.float16))
rep("q.to(fp32)                       ", lambda: q.to(torch.float32))

print("\n--- 2) 反量化组合 ---")


def v_current():
    w = q.to(torch.bfloat16).view(E_N, N, K // gs, gs)
    return (w * s.to(torch.bfloat16).unsqueeze(-1)).reshape(E_N, N, K)


def v_fp32mul():
    w = q.float().view(E_N, N, K // gs, gs)
    return (w * s.float().unsqueeze(-1)).reshape(E_N, N, K).to(torch.bfloat16)


def v_fp16mul():
    w = q.to(torch.float16).view(E_N, N, K // gs, gs)
    return (w * s.unsqueeze(-1)).reshape(E_N, N, K)


def v_scale_expand():
    sf = s.to(torch.bfloat16).repeat_interleave(gs, dim=-1)
    return q.to(torch.bfloat16) * sf


def v_view_then_mul():
    w = q.view(E_N, N, K // gs, gs).to(torch.bfloat16)
    return w * s.to(torch.bfloat16).unsqueeze(-1)


rep("当前 deq（bf16 cast → mul）       ", v_current)
rep("fp32 mul → bf16                  ", v_fp32mul, extra_out=q.numel() * 2)
rep("fp16 mul（输出 fp16，4B/组→2B 写）", v_fp16mul)
rep("scale repeat_interleave → mul    ", v_scale_expand, extra_out=q.numel() * 2)
rep("先 view 再 cast 再 mul           ", v_view_then_mul)

print("\n--- 3) 整条链：gather + deq + bmm ---")


def full_current():
    wgu = v_current()
    return torch.bmm(x, wgu.transpose(1, 2))


def full_fp16():
    w = v_fp16mul()
    return torch.bmm(x.to(torch.float16), w.transpose(1, 2))


rep("gather(隐) + bf16 deq + bmm      ", full_current)
rep("gather(隐) + fp16 deq + bmm fp16 ", full_fp16)

print("\n--- 4) 打包路径（4bit → 解包 → deq）---")
qp = E.pack4(q)
print(f"  打包后 {tuple(qp.shape)} = {qp.numel()/1e6:.1f}MB")


class _Pack:
    pass


def unpack_bf16():
    p16 = qp.to(torch.int16) & 0xFF
    lo = ((p16 & 0x0F) ^ 8) - 8
    hi = (((p16 >> 4) & 0x0F) ^ 8) - 8
    qq = torch.stack([lo, hi], dim=-1).reshape(E_N, N, K).to(torch.bfloat16)
    w = qq.view(E_N, N, K // gs, gs)
    return (w * s.to(torch.bfloat16).unsqueeze(-1)).reshape(E_N, N, K)


rep("unpack(int16) → bf16 → deq       ", unpack_bf16, extra_out=q.numel() * 2)

print("\n--- 5) 稠密 W8 单层总量参考（每层 77.3M 参数，bf16 常驻直读）---")
wd = torch.randn(10240, 2560, dtype=torch.bfloat16, device=DEV) / 50
xx = torch.randn(1, 2560, dtype=torch.bfloat16, device=DEV)
t = bench(lambda: torch.nn.functional.linear(xx, wd))
print(f"  bf16 直读 GEMV [2560→10240]      {t:7.3f} ms   {wd.numel()*2/1e9/(t/1e3):7.1f} GB/s")
