"""p326: 新 MoE 的图捕获正确性（生产口径：输出 copy_ 到常驻缓冲区）。"""
import time
import traceback

import torch
import torch_npu

DEV = "npu:0"
torch.manual_seed(0)
F = torch_npu.npu_weight_quant_batchmatmul
pack = torch_npu.npu_convert_weight_to_int4pack
gs = 64
H, MI, TOPK = 2560, 640, 10
G_GU, N_GU, G_DN, N_DN = H // gs, 2 * MI, MI // gs, H
E = 512


def build(K, N):
    G = K // gs
    W = torch.randn(E, N, K) * 0.02
    s = (W.abs().view(E, N, G, gs).amax(-1) / 7.0 + 1e-9).float()
    q = torch.round(W / s.unsqueeze(-1).repeat(1, 1, 1, gs).view(E, N, K)).clamp_(-8, 7).to(torch.int32)
    art = torch.empty(E, K, N // 8, dtype=torch.int32, device=DEV)
    for e in range(E):
        art[e] = pack(q[e].t().contiguous().to(DEV))
    del W, q
    base = (s.amax(-1) / 127.0).to(torch.float16)
    s8 = torch.round(s / base.float().unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
    return dict(art_t=art.transpose(0, 1).contiguous(),
                s8t=s8.permute(2, 0, 1).contiguous().to(DEV),
                base=base.to(torch.bfloat16).to(DEV),
                K=K, N=N)


t0 = time.perf_counter()
GU, DN = build(H, N_GU), build(MI, N_DN)
torch.npu.synchronize()
print("build %.0fs" % (time.perf_counter() - t0))
print("HBM free %.1f GiB" % (torch.npu.mem_get_info()[0] / 2**30))

x = torch.randn(1, H, dtype=torch.bfloat16, device=DEV)
ids = torch.arange(TOPK, dtype=torch.int64, device=DEV)
w_out = torch.softmax(torch.randn(TOPK, device=DEV), 0).to(torch.bfloat16)
out = torch.zeros(1, H, dtype=torch.bfloat16, device=DEV)


def catw(A, ids):
    p = torch.index_select(A["art_t"], 1, ids)          # [K,k,N/8]
    return p.reshape(A["K"], p.shape[1] * (A["N"] // 8))


def cats(A, ids, G):
    sc = torch.index_select(A["s8t"], 1, ids)           # [G,k,N] int8
    bs = torch.index_select(A["base"], 0, ids).unsqueeze(0)
    return (sc.to(torch.bfloat16) * bs).reshape(G, -1)


def moe(x, ids, w_out, out):
    y = F(x, catw(GU, ids), cats(GU, ids, G_GU), antiquant_group_size=gs).view(TOPK, 2 * MI)
    act = (torch.nn.functional.silu(y[:, :MI].float()) * y[:, MI:].float()).to(torch.bfloat16)
    y2 = F(act, catw(DN, ids), cats(DN, ids, G_DN), antiquant_group_size=gs).view(TOPK, TOPK, H)
    d = torch.arange(TOPK, device=DEV)
    r = (y2[d, d] * w_out.unsqueeze(-1)).sum(0, keepdim=True)
    out.copy_(r)
    return out


def bench(fn, n=20):
    for _ in range(4):
        fn()
    torch.npu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


print("eager MoE %.3f ms（当前生产 1.00 ms/层）" % bench(lambda: moe(x, ids, w_out, out)))
ref = moe(x, ids, w_out, out).clone()
try:
    st = torch.npu.Stream()
    with torch.npu.stream(st):
        for _ in range(3):
            moe(x, ids, w_out, out)
        torch.npu.synchronize()
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g, capture_error_mode="relaxed"):
            moe(x, ids, w_out, out)
        torch.npu.synchronize()
        out.zero_()
        g.replay()
        torch.npu.synchronize()
        got = out.clone()
        rel = float((got.float() - ref.float()).abs().mean() / (ref.float().abs().mean() + 1e-9))
        print("捕获后 replay vs eager rel %.3e -> %s" % (rel, "PASS" if rel < 1e-3 else "FAIL"))
        ids.copy_(torch.randperm(E, device=DEV)[:TOPK].to(torch.int64))
        ref2 = moe(x, ids, w_out, out).clone()
        out.zero_()
        g.replay()
        torch.npu.synchronize()
        got2 = out.clone()
        rel2 = float((got2.float() - ref2.float()).abs().mean() / (ref2.float().abs().mean() + 1e-9))
        print("换 ids（422 号等）后 rel %.3e -> %s" % (rel2, "PASS" if rel2 < 1e-3 else "FAIL"))
        print("  replay %.3f ms / eager %.3f ms" % (bench(lambda: g.replay()),
                                                    bench(lambda: moe(x, ids, w_out, out))))
        # 连续两次 replay 是否稳定（模拟多 token）
        for _ in range(3):
            g.replay()
        torch.npu.synchronize()
        print("  连续 replay 后 finite:", bool(torch.isfinite(out).all()))
except Exception:  # noqa: BLE001
    print("捕获 FAIL", [ln for ln in traceback.format_exc().splitlines() if ln.strip()][-1][:400])
