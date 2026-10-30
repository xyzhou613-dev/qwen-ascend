"""W4A16 GEMV 探针：① 启动开销 ② 4-bit 解包+反量化+GEMV 的正确性与吞吐。"""
import sys, time
import torch
import torch_npu  # noqa
import triton
import triton.language as tl

sys.path.insert(0, "/root/ascend-dev/fnx")
from fnx import engine as E
from fnx.load import load_packed_experts, load_layer

DEV = "npu:0"


@triton.jit
def nop_kernel(Y, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    off = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + off, tl.zeros((BLOCK,), tl.float32))


@triton.jit
def w4_gemv_kernel(X, W, S, Y, K,
                   sx_b, sw_b, sw_n, ss_b, ss_n, sy_b,
                   BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, GS: tl.constexpr):
    """y[b,n] = sum_k unpack(W[b,n,k//2], k) * s[b,n,k//GS] * x[b,k]"""
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, BLOCK_K // 2)
    offs_g = tl.arange(0, BLOCK_K // GS)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    xp = X + pid_b * sx_b
    wp = W + pid_b * sw_b + offs_n[:, None] * sw_n + offs_h[None, :]
    sp = S + pid_b * ss_b + offs_n[:, None] * ss_n + offs_g[None, :]
    for k0 in range(0, K, BLOCK_K):
        p = tl.load(wp + k0 // 2).to(tl.int16)
        lo = ((p & 0x0F) ^ 8) - 8
        hi = (((p >> 4) & 0x0F) ^ 8) - 8
        w4 = tl.reshape(tl.join(lo, hi), (BLOCK_N, BLOCK_K))
        s = tl.load(sp + k0 // GS).to(tl.float32)
        sb = tl.reshape(tl.broadcast_to(tl.reshape(s, (BLOCK_N, BLOCK_K // GS, 1)),
                                        (BLOCK_N, BLOCK_K // GS, GS)), (BLOCK_N, BLOCK_K))
        w = w4.to(tl.float32) * sb
        x = tl.load(xp + k0 + tl.arange(0, BLOCK_K)).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(Y + pid_b * sy_b + offs_n, acc.to(tl.bfloat16))


def bench(fn, warm=5, n=20):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    print("=== Q1 启动开销 ===", flush=True)
    buf = torch.zeros(1 << 20, dtype=torch.float32, device=DEV)
    for nprog in (1, 10, 100, 800):
        t = bench(lambda: nop_kernel[(nprog,)](buf, BLOCK=32))
        print(f"  num_programs={nprog:5d}  {t*1e3:7.2f} us/kernel", flush=True)

    print("\n=== Q2 W4A16 GEMV（真实形状，B=10 专家）===", flush=True)
    pk = load_packed_experts(0, DEV)
    W0 = load_layer("/models/Qwen3.8-Flash-Next-Q/L00.safetensors", 0, DEV)
    idx = torch.tensor([7, 343, 105, 212, 243, 254, 403, 168, 302, 67], device=DEV)
    torch.manual_seed(0)

    for which, key, N, K in (("gate_up_proj", "mlp.experts.gate_up_proj", 1280, 2560),
                             ("down_proj", "mlp.experts.down_proj", 2560, 640)):
        p4, s4 = pk[which]
        B = 10
        wb = p4.index_select(0, idx).contiguous()
        sb = s4.index_select(0, idx).contiguous()
        x = (torch.randn(B, K, dtype=torch.bfloat16, device=DEV) / 16)
        y = torch.zeros(B, N, dtype=torch.bfloat16, device=DEV)

        w_ref = W0[key].gather_deq(idx).float()
        ref = torch.bmm(x.float().unsqueeze(1), w_ref.transpose(1, 2)).squeeze(1)

        ok = None
        for BLOCK_N, BLOCK_K in ((32, 64), (32, 32), (16, 64), (64, 32)):
            grid = (N // BLOCK_N, B)

            def run():
                w4_gemv_kernel[grid](x, wb, sb, y, K,
                                     x.stride(0), wb.stride(0), wb.stride(1),
                                     sb.stride(0), sb.stride(1), y.stride(0),
                                     BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GS=32)
            try:
                run()
                torch.npu.synchronize()
                ok = (BLOCK_N, BLOCK_K, grid, run)
                print(f"  {which:14s} 可用配置 BLOCK_N={BLOCK_N} BLOCK_K={BLOCK_K} grid={grid}", flush=True)
                break
            except Exception as ex:
                print(f"  {which:14s} BLOCK_N={BLOCK_N} BLOCK_K={BLOCK_K} 编译失败: {str(ex)[:120]}", flush=True)
        if ok is None:
            print(f"  >>> {which} 所有配置都编不过", flush=True)
            continue
        BLOCK_N, BLOCK_K, grid, run = ok
        d = (y.float() - ref).abs()
        rel = d.mean().item() / ref.abs().mean().clamp_min(1e-9).item()
        print(f"  {which:14s} N={N} K={K} grid={grid} maxdiff {d.max().item():.3e} rel {rel*100:.4f}%", flush=True)

        t = bench(run)
        gb = wb.numel() / 1e9
        print(f"                 {t:7.3f} ms  {gb/(t/1e3):7.0f} GB/s(packed)  → 96 段合计 {t*96:6.1f} ms", flush=True)

    print("\n=== 对照：torch 未打包 int8 路径 ===", flush=True)
    eq = W0["mlp.experts.gate_up_proj"]
    t = bench(lambda: eq.gather_deq(idx))
    print(f"  gather_deq gate_up  {t:7.3f} ms → 48 层 {t*48:6.1f} ms", flush=True)


if __name__ == "__main__":
    main()
