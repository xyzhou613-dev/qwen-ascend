import json, time
import numpy as np, torch, torch_npu
from fnx.w4gemv_op import W4GemvOp, GS
GOLD = "/root/ascend-dev/golden/w4gemv"
meta = json.load(open(GOLD + "/meta.json")); c = meta["cases"]["multi_gate_up_proj"]
nE, N, K = c["B"], c["N"], c["K"]
p4 = np.fromfile(f"{GOLD}/multi/gate_up_proj_p4.bin", dtype=np.int8).reshape(nE, N, K//2)
s4 = np.fromfile(f"{GOLD}/multi/gate_up_proj_s4.bin", dtype=np.float16).reshape(nE, N, K//GS)
x  = ((np.fromfile(f"{GOLD}/multi/gate_up_proj_x.bin", dtype=np.int16).astype(np.uint32)<<16).view(np.float32)).reshape(nE,K)
op = W4GemvOp()
s4t = op.prepare_layer(s4)
p4v2 = torch.from_numpy(np.ascontiguousarray(op.to_v2_layout(p4,N,K))).npu()
xdev = op.make_x_dev(x, lay=2)
NL = 24
def bench(fn, rep=10):
    for _ in range(3): fn()
    torch.npu.synchronize(); t0=time.perf_counter()
    for _ in range(rep): fn()
    torch.npu.synchronize(); return (time.perf_counter()-t0)/rep*1e3
t_single = bench(lambda: [op.run(p4v2, s4t, xdev, nE, N, K, lay=2) for _ in range(NL)])
ys = [torch.empty((nE,N), dtype=torch.float32).npu() for _ in range(NL)]
t_batch = bench(lambda: op.run_layers([p4v2]*NL, [s4t]*NL, [xdev]*NL, ys, N, K, nE, lay=2))
print(f"24 层逐个 run(): {t_single:.3f} ms   一次 run_layers(): {t_batch:.3f} ms   省 {100*(t_single-t_batch)/t_single:.1f}%")
print(f"外推 48 层/卡: 逐个 {t_single*2:.2f} ms  批量 {t_batch*2:.2f} ms")