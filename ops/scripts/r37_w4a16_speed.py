import time
import traceback

import torch
import torch_npu

DEV = "npu:0"
torch.manual_seed(0)
f = torch_npu.npu_weight_quant_batchmatmul
pack = torch_npu.npu_convert_weight_to_int4pack
free, total = torch.npu.mem_get_info()
print("HBM free/total: %.2f / %.2f GiB" % (free / 2**30, total / 2**30))

gu = dict(K=2560, N=1280)      # gate_up: 每个专家 [K=2560 -> N=1280]
dn = dict(K=640, N=2560)       # down:  每个专家 [K=640  -> N=2560]
E = 64                          # 只建 64 个专家（每 token 只读 10 个，时序代表）
gs = 64


def build(cfg, E):
    K, N = cfg["K"], cfg["N"]
    G = K // gs
    W = torch.randn(E, N, K) * 0.02
    s = (W.abs().view(E, N, G, gs).amax(-1) / 7.0 + 1e-9).float()
    q = torch.round(W / s.unsqueeze(-1).repeat(1, 1, 1, gs).view(E, N, K)).clamp_(-8, 7).to(torch.int32)
    wdeq = (q.view(E, N, G, gs) * s.unsqueeze(-1)).reshape(E, N, K).to(torch.bfloat16)
    art = torch.empty(E, K, N // 8, dtype=torch.int32, device=DEV)
    for e in range(E):
        art[e] = pack(q[e].t().contiguous().to(DEV))
    return art, s.to(torch.bfloat16).to(DEV), wdeq


t0 = time.perf_counter()
art_gu, s_gu, w_gu = build(gu, E)      # art_gu [E,K,N/8]  [E,N,G] bf16  [E,N,K] bf16
art_dn, s_dn, w_dn = build(dn, E)
torch.npu.synchronize()
print("build %.1fs  art_gu %s %.0f MB  art_dn %s %.0f MB" % (
    time.perf_counter() - t0, tuple(art_gu.shape), art_gu.numel() * 4 / 2**20,
    tuple(art_dn.shape), art_dn.numel() * 4 / 2**20))

x = torch.randn(1, gu["K"], dtype=torch.bfloat16, device=DEV)
ids = torch.arange(10, dtype=torch.int64, device=DEV)


def bench(fn, n=30):
    for _ in range(5):
        fn()
    torch.npu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def relerr(y, ref):
    d = (y.float().cpu() - ref.float()).abs()
    return float(d.max()), float(d.mean() / (ref.abs().mean() + 1e-9))


# ---------- 官方标准调用：单专家 ----------
s1 = s_gu[0].t().contiguous()          # [G,N]
ref1 = x.float().cpu() @ w_gu[0].float().t()
try:
    y = f(x, art_gu[0], s1, antiquant_group_size=gs)
    mx, rl = relerr(y, ref1)
    print("单专家 M=1 数值: max %.3e rel %.3e" % (mx, rl))
except Exception:  # noqa: BLE001
    print("单专家 FAIL", traceback.format_exc().splitlines()[-1][:180])

print("\n=== 时序 A: 10 次单专家调用（模仿逐专家） ===")
sc10 = [s_gu[e].t().contiguous() for e in range(10)]
t_a = bench(lambda: [f(x, art_gu[e], sc10[e], antiquant_group_size=gs) for e in range(10)])
print("  10x 单专家 gu            %.3f ms  (当前 torch 路径一层 MoE ≈ 1.00 ms)" % t_a)

print("\n=== 时序 B: 拼接成一次调用（K=2560, N=10*1280） ===")


def cat_call(art, ids, sc_all):
    p = torch.index_select(art, 0, ids)                 # [10,K,N/8]
    w = p.permute(1, 0, 2).reshape(p.shape[1], -1)      # [K,10*N/8] 连续
    sc = torch.index_select(sc_all, 0, ids)             # [10,N,G]
    scc = sc.permute(1, 0, 2).reshape(sc.shape[1], -1).contiguous().to(torch.bfloat16)  # [G,n*N]
    return w, scc


sc_all_gu = s_gu.transpose(1, 2).contiguous()           # [E,G,N]
try:
    w_cat, sc_cat = cat_call(art_gu, ids, sc_all_gu)
    y = f(x, w_cat, sc_cat, antiquant_group_size=gs)
    ref_cat = torch.cat([(x.float().cpu() @ w_gu[e].float().t()) for e in range(10)], dim=1)
    mx, rl = relerr(y, ref_cat)
    print("  数值: %s max %.3e rel %.3e" % (tuple(y.shape), mx, rl))
    t_b = bench(lambda: f(x, *(cat_call(art_gu, ids, sc_all_gu)), antiquant_group_size=gs))
    print("  gather+permute+1 call     %.3f ms" % t_b)
    w_cat, sc_cat = cat_call(art_gu, ids, sc_all_gu)
    t_b2 = bench(lambda: f(x, w_cat, sc_cat, antiquant_group_size=gs))
    print("  纯 matmul（省掉拼接）      %.3f ms" % t_b2)
    t_b3 = bench(lambda: cat_call(art_gu, ids, sc_all_gu))
    print("  只拼接（gather+permute）   %.3f ms" % t_b3)
except Exception:  # noqa: BLE001
    print("  拼接 FAIL", [ln for ln in traceback.format_exc().splitlines() if ln.strip()][-1][:250])

print("\n=== 时序 C: down_proj 拼接（K=640, N=10*2560） ===")
try:
    xd = torch.randn(1, dn["K"], dtype=torch.bfloat16, device=DEV)
    sc_all_dn = s_dn.transpose(1, 2).contiguous()
    wd_cat, sc_dn_cat = cat_call(art_dn, ids, sc_all_dn)
    yd = f(xd, wd_cat, sc_dn_cat, antiquant_group_size=gs)
    ref_d = torch.cat([(xd.float().cpu() @ w_dn[e].float().t()) for e in range(10)], dim=1)
    mx, rl = relerr(yd, ref_d)
    print("  数值: %s max %.3e rel %.3e" % (tuple(yd.shape), mx, rl))
    t_c = bench(lambda: f(xd, *(cat_call(art_dn, ids, sc_all_dn)), antiquant_group_size=gs))
    print("  gather+permute+1 call     %.3f ms" % t_c)
except Exception:  # noqa: BLE001
    print("  down FAIL", [ln for ln in traceback.format_exc().splitlines() if ln.strip()][-1][:250])

print("\n=== 时序 D: NPUGraph 捕获是否可行 ===")
try:
    st = torch.npu.Stream()
    with torch.npu.stream(st):
        for _ in range(3):
            f(x, *(cat_call(art_gu, ids, sc_all_gu)), antiquant_group_size=gs)
        torch.npu.synchronize()
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g, capture_error_mode="relaxed"):
            y_cap = f(x, *(cat_call(art_gu, ids, sc_all_gu)), antiquant_group_size=gs)
        torch.npu.synchronize()
        y_eager = f(x, *(cat_call(art_gu, ids, sc_all_gu)), antiquant_group_size=gs)
        g.replay()
        torch.npu.synchronize()
        d = (y_cap.float() - y_eager.float()).abs().mean() / (y_eager.float().abs().mean() + 1e-9)
        print("  捕获 OK，replay vs eager rel %.3e  ->  %s"
              % (float(d), "PASS" if float(d) < 1e-3 else "FAIL"))
        t_g = bench(lambda: g.replay())
        print("  graph replay             %.3f ms" % t_g)
except Exception:  # noqa: BLE001
    print("  捕获 FAIL", [ln for ln in traceback.format_exc().splitlines() if ln.strip()][-1][:250])
