"""热态对比：打包 vs 未打包 的 gather+反量化 真实耗时。

第 25 轮的 "84 ms vs 0.55 ms（慢 150x）" 是一次冷调用对一次热循环，口径不可比。
本脚本在同一进程、同一形状、同样 warmup + 20 次循环下比较，并给出整层 MoE 成本。
"""
import sys, time
import torch
sys.path.insert(0, "/root/ascend-dev/fnx")
import torch_npu  # noqa
from fnx import engine as E

DEV = "npu:0"
torch.manual_seed(0)


def bench(fn, warm=5, n=20):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def make(shapes):
    """shapes: [(N,K)], 返回 (packed_ExpQ, unpacked_ExpQ) 列表 + idx"""
    pk, up = [], []
    for (N, K) in shapes:
        q = torch.randint(-8, 8, (E_TOPK, N, K), dtype=torch.int8, device=DEV)
        s = (torch.rand(E_TOPK, N, K // 32, dtype=torch.float32, device=DEV) * 0.01).to(torch.float16)
        up.append(E.ExpQ(q, s, 32, packed=False))
        a = E.ExpQ(q, s, 32, packed=False)
        a.q = E.pack4(q)
        a.K = K
        a.packed = True
        pk.append(a)
    return pk, up


E_TOPK = 10
SHAPES = [(1280, 2560), (2560, 640)]   # gate_up_proj, down_proj（单个专家）
pk, up = make(SHAPES)
idx = torch.arange(E_TOPK, device=DEV)
x = torch.randn(E_TOPK, 1, 2560, dtype=torch.bfloat16, device=DEV) / 20

# 正确性
w_pk = pk[0].gather_deq(idx)
w_up = up[0].gather_deq(idx)
print("打包 vs 未打包 反量化结果 torch.equal:", torch.equal(w_pk, w_up), flush=True)

print("\n--- 单项：gather_deq（10 专家）---")
for tag, lst in (("未打包 int8", up), ("真4bit 打包", pk)):
    t0 = bench(lambda: lst[0].gather_deq(idx))
    t1 = bench(lambda: lst[1].gather_deq(idx))
    tp = t0 + t1
    print(f"  {tag:14s} gate_up {t0:6.3f} ms  down {t1:6.3f} ms  单层合计 {tp:6.3f} ms "
          f"→ 48 层 {tp*48:7.1f} ms")

print("\n--- 单项：bmm ---")


def moe_packed():
    wgu = pk[0].gather_deq(idx)
    y = torch.bmm(x, wgu.transpose(1, 2)).squeeze(1)
    act = (torch.nn.functional.silu(y[:, :640].float()) * y[:, 640:].float()).to(torch.bfloat16)
    wdn = pk[1].gather_deq(idx)
    y2 = torch.bmm(act.unsqueeze(1), wdn.transpose(1, 2)).squeeze(1)
    return y2.sum(0)


def moe_unpacked():
    wgu = up[0].gather_deq(idx)
    y = torch.bmm(x, wgu.transpose(1, 2)).squeeze(1)
    act = (torch.nn.functional.silu(y[:, :640].float()) * y[:, 640:].float()).to(torch.bfloat16)
    wdn = up[1].gather_deq(idx)
    y2 = torch.bmm(act.unsqueeze(1), wdn.transpose(1, 2)).squeeze(1)
    return y2.sum(0)


for tag, fn in (("未打包 int8", moe_unpacked), ("真4bit 打包", moe_packed)):
    t = bench(fn)
    print(f"  专家整段（gather+deq+2xbmm）{tag:14s} {t:6.3f} ms/层 → 48 层 {t*48:7.1f} ms")

print("\n--- 参考：稠密 bf16 GEMV（每层 77.3M 参数）---")
wqkv = torch.randn(10240, 2560, dtype=torch.bfloat16, device=DEV) / 50
h = torch.randn(1, 2560, dtype=torch.bfloat16, device=DEV)
t = bench(lambda: torch.nn.functional.linear(h, wqkv))
print(f"  in_proj_qkv bf16 [2560→10240]  {t:6.3f} ms  {wqkv.numel()*2/1e9/(t/1e3):6.1f} GB/s")

print("\n--- 参考：反量化 int8→bf16 的纯转换带宽 ---")
q = torch.randint(-8, 8, (10, 1280, 2560), dtype=torch.int8, device=DEV)
t = bench(lambda: q.to(torch.bfloat16))
print(f"  q.to(bf16) 32.8M 元  {t:6.3f} ms  {q.numel()*3/1e9/(t/1e3):6.1f} GB/s (读1B+写2B)")
print("OK")
