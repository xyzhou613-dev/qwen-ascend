"""p325: 新 MoE（per-group W4A16 aclnn）各环节最优形态 + 图捕获。

E=16 做数值；E=512 做时序（真实 artifact 分布，避免 L2 假快）。
"""
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
G_GU, N_GU = H // gs, 2 * MI        # 40, 1280
G_DN, N_DN = MI // gs, H            # 10, 2560

free, total = torch.npu.mem_get_info()
print("HBM free/total: %.1f / %.1f GiB" % (free / 2**30, total / 2**30))


def build(K, N, E):
    G = K // gs
    W = torch.randn(E, N, K) * 0.02
    s = (W.abs().view(E, N, G, gs).amax(-1) / 7.0 + 1e-9).float()
    q = torch.round(W / s.unsqueeze(-1).repeat(1, 1, 1, gs).view(E, N, K)).clamp_(-8, 7).to(torch.int32)
    wdeq = (q.view(E, N, G, gs) * s.unsqueeze(-1)).reshape(E, N, K).to(torch.bfloat16)
    art = torch.empty(E, K, N // 8, dtype=torch.int32, device=DEV)
    for e in range(E):
        art[e] = pack(q[e].t().contiguous().to(DEV))
    art_t = art.transpose(0, 1).contiguous()                   # [K,E,N/8]
    base_bf = (s.amax(-1) / 127.0).to(torch.float16).to(torch.bfloat16)
    s8 = torch.round(s / base_bf.float().unsqueeze(-1)).clamp_(-127, 127).to(torch.int8)
    s8t = s8.permute(2, 0, 1).contiguous()                     # [G,E,N]
    sbf = (s8t.float() * base_bf.unsqueeze(0).float()); assert sbf.shape == s8t.shape
    sbf = sbf.to(torch.bfloat16).to(DEV).contiguous()          # [G,E,N] 已乘 base
    return dict(art=art, art_t=art_t, s8=s8.to(DEV), s8t=s8t.to(DEV), sbf=sbf,
                base=base_bf.to(DEV), wdeq=wdeq)


def bench(fn, n=20):
    for _ in range(4):
        fn()
    torch.npu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


# ---------------- 权重拼接两种布局 ----------------
def catw_perm(A, ids, N8):
    p = torch.index_select(A["art"], 0, ids)               # [k,K,N/8]
    return p.permute(1, 0, 2).reshape(A["art"].shape[1], p.shape[0] * N8).contiguous()


def catw_t(A, ids, N8):
    p = torch.index_select(A["art_t"], 1, ids)             # [K,k,N/8] 连续
    return p.reshape(A["art_t"].shape[0], p.shape[1] * N8)


# ---------------- scale 拼接三种 ----------------
def cats_v1(A, ids, G):
    sc = torch.index_select(A["s8"], 0, ids)               # [k,N,G]
    bs = torch.index_select(A["base"], 0, ids)             # [k,N]
    bf = (sc.float() * bs.float().unsqueeze(-1)).to(torch.bfloat16)
    return bf.permute(2, 0, 1).reshape(G, -1).contiguous()


def cats_v2(A, ids, G):
    sc = torch.index_select(A["s8t"], 1, ids)              # [G,k,N] int8
    bs = torch.index_select(A["base"], 0, ids).unsqueeze(0)  # [1,k,N]
    bf = sc.to(torch.bfloat16) * bs
    return bf.reshape(G, -1)


def cats_v3(A, ids, G):
    return torch.index_select(A["sbf"], 1, ids).reshape(G, -1)


print("\n" + "=" * 70)
print("A) 数值（E=16）")
num = {"gu": build(H, N_GU, 16), "dn": build(MI, N_DN, 16)}
x = torch.randn(1, H, dtype=torch.bfloat16, device=DEV)
ids = torch.arange(TOPK, dtype=torch.int64, device=DEV)
w_out = torch.softmax(torch.randn(TOPK, device=DEV), 0).to(torch.bfloat16)


def run_moe(num, ids, w_out, catw=catw_perm, cats=cats_v3):
    A = num["gu"]
    N8 = N_GU // 8
    w = catw(A, ids, N8)
    s = cats(A, ids, G_GU)
    y = F(x, w, s, antiquant_group_size=gs).view(TOPK, 2 * MI)
    act = (torch.nn.functional.silu(y[:, :MI].float()) * y[:, MI:].float()).to(torch.bfloat16)
    B = num["dn"]
    w2 = catw(B, ids, N_DN // 8)
    s2 = cats(B, ids, G_DN)
    y2 = F(act, w2, s2, antiquant_group_size=gs).view(TOPK, TOPK, H)
    d = torch.arange(TOPK, device=DEV)
    out = y2[d, d]
    return (out * w_out.unsqueeze(-1)).sum(0, keepdim=True)


try:
    xc = x.cpu().float()
    wg, wd = num["gu"]["wdeq"], num["dn"]["wdeq"]
    y_gu = torch.stack([xc[0] @ wg[e].float().t() for e in range(TOPK)])
    act_r = (torch.nn.functional.silu(y_gu[:, :MI]) * y_gu[:, MI:]).to(torch.bfloat16)
    y_dn = torch.stack([act_r[e].float() @ wd[e].float().t() for e in range(TOPK)])
    y_ref = (y_dn * w_out.cpu().float().unsqueeze(-1)).sum(0, keepdim=True)
    for tag, kw in (("v1/perm", dict(cats=cats_v1)), ("v2/perm", dict(cats=cats_v2)),
                    ("v3/perm", dict(cats=cats_v3)), ("v3/art_t", dict(catw=catw_t, cats=cats_v3)),
                    ("v2/art_t", dict(catw=catw_t, cats=cats_v2))):
        y = run_moe(num, ids, w_out, **kw).float().cpu()
        d = (y - y_ref).abs()
        rl = float(d.mean() / (y_ref.abs().mean() + 1e-9))
        print("  %-10s max %.3e rel %.3e  %s" % (tag, float(d.max()), rl, "PASS" if rl < 3e-2 else "FAIL"))
except Exception:  # noqa: BLE001
    print("  FAIL", [ln for ln in traceback.format_exc().splitlines() if ln.strip()][-1][:300])

del num
torch.npu.empty_cache()

print("\n" + "=" * 70)
print("B) 时序（E=512，真实分布）")
t0 = time.perf_counter()
big = {"gu": build(H, N_GU, 512), "dn": build(MI, N_DN, 512)}
big["gu"].pop("wdeq", None)
big["dn"].pop("wdeq", None)
torch.npu.synchronize()
print("  build %.1fs  art_gu %.0f MB  art_dn %.0f MB  sbf_gu %.0f MB  sbf_dn %.0f MB" % (
    time.perf_counter() - t0,
    big["gu"]["art"].numel() * 4 / 2**20, big["dn"]["art"].numel() * 4 / 2**20,
    big["gu"]["sbf"].numel() * 2 / 2**20, big["dn"]["sbf"].numel() * 2 / 2**20))
E512 = 512
ids5 = torch.randperm(E512, device=DEV)[:TOPK].to(torch.int64)

for tag, cw, cs in (("perm + v1", catw_perm, cats_v1), ("perm + v2", catw_perm, cats_v2),
                    ("perm + v3", catw_perm, cats_v3), ("art_t + v2", catw_t, cats_v2),
                    ("art_t + v3", catw_t, cats_v3)):
    A = big["gu"]
    t_cw = bench(lambda: cw(A, ids5, N_GU // 8))
    t_cs = bench(lambda: cs(A, ids5, G_GU))
    w = cw(A, ids5, N_GU // 8)
    s = cs(A, ids5, G_GU)
    t_call = bench(lambda: F(x, w, s, antiquant_group_size=gs))
    t_call1 = bench(lambda: F(x, w, s, antiquant_group_size=gs, inner_precise=1))
    print("  %-11s 权重拼接 %.3f  scale拼接 %.3f  gu调用 %.3f (ip1 %.3f)  合计 %.3f ms"
          % (tag, t_cw, t_cs, t_call, t_call1, t_cw + t_cs + t_call))

print("\n  --- dn（K=640,N=2560）---")
for tag, cw, cs in (("art_t + v2", catw_t, cats_v2), ("art_t + v3", catw_t, cats_v3)):
    B = big["dn"]
    act = torch.randn(TOPK, MI, dtype=torch.bfloat16, device=DEV)
    t_cw = bench(lambda: cw(B, ids5, N_DN // 8))
    t_cs = bench(lambda: cs(B, ids5, G_DN))
    w2 = cw(B, ids5, N_DN // 8)
    s2 = cs(B, ids5, G_DN)
    t_call = bench(lambda: F(act, w2, s2, antiquant_group_size=gs))
    print("  %-11s 权重拼接 %.3f  scale拼接 %.3f  M=10 对角法调用 %.3f  合计 %.3f ms"
          % (tag, t_cw, t_cs, t_call, t_cw + t_cs + t_call))
    # 对照：10 次单专家调用
    wl = [torch.index_select(B["art"], 0, ids5)[e] for e in range(TOPK)]
    sl = [cs(B, ids5, G_DN)[:, e * N_DN:(e + 1) * N_DN].contiguous() for e in range(TOPK)]
    t10 = bench(lambda: [F(act[e:e + 1], wl[e], sl[e], antiquant_group_size=gs) for e in range(TOPK)])
    print("                对照 10 次单专家调用            %.3f ms" % t10)

print("\n" + "=" * 70)
print("C) 图捕获 + 动态 ids")
try:
    st = torch.npu.Stream()
    fn = lambda: run_moe(big, ids5, w_out, catw=catw_t, cats=cats_v2)  # noqa: E731
    with torch.npu.stream(st):
        for _ in range(3):
            fn()
        torch.npu.synchronize()
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g, capture_error_mode="relaxed"):
            y_cap = fn()
        torch.npu.synchronize()
        y_ea = fn().clone()
        y_cap = y_cap.clone()
        g.replay()
        torch.npu.synchronize()
        rel = float((y_cap.float() - y_ea.float()).abs().mean() / (y_ea.float().abs().mean() + 1e-9))
        print("  捕获 OK  replay vs eager rel %.3e -> %s" % (rel, "PASS" if rel < 1e-3 else "FAIL"))
        ids5.copy_(torch.randperm(E512, device=DEV)[:TOPK].to(torch.int64))
        g.replay()
        torch.npu.synchronize()
        y_ea2 = fn().clone()
        rel2 = float((y_cap.float() - y_ea2.float()).abs().mean() / (y_ea2.float().abs().mean() + 1e-9))
        print("  换 ids 后 replay vs eager rel %.3e -> %s" % (rel2, "PASS" if rel2 < 1e-3 else "FAIL"))
        print("  graph replay %.3f ms（eager %.3f ms）" % (bench(lambda: g.replay()), bench(fn)))
except Exception:  # noqa: BLE001
    print("  捕获 FAIL", [ln for ln in traceback.format_exc().splitlines() if ln.strip()][-1][:400])
