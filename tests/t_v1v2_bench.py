"""v1 vs v2 布局耗时对照（同一份数据、同一进程、热态）。

用途：验证"v2 省掉 x 拆分 + 归约减半"这两条推断。
"""
import json
import time

import numpy as np
import torch
import torch_npu  # noqa: F401

from fnx.w4gemv_op import W4GemvOp, GS

GOLD = "/root/ascend-dev/golden/w4gemv"


def load(name):
    meta = json.load(open(GOLD + "/meta.json"))
    c = meta["cases"][name]
    base = name.replace("multi_", "")
    d = GOLD + "/" + c["dir"]
    nE, N, K = c["B"], c["N"], c["K"]
    p4 = np.fromfile(f"{d}/{base}_p4.bin", dtype=np.int8).reshape(nE, N, K // 2)
    s4 = np.fromfile(f"{d}/{base}_s4.bin", dtype=np.float16).reshape(nE, N, K // GS)
    x = ((np.fromfile(f"{d}/{base}_x.bin", dtype=np.int16).astype(np.uint32) << 16)
         .view(np.float32)).reshape(nE, K)
    return p4, s4, x, nE, N, K


def bench(fn, warm=3, rep=20):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(rep):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / rep * 1e3


def main():
    op = W4GemvOp()
    for name in ("multi_gate_up_proj", "multi_down_proj"):
        p4, s4, x, nE, N, K = load(name)
        s4t = op.prepare_layer(s4)
        p4_dev = torch.from_numpy(p4).npu()
        p4v2_dev = torch.from_numpy(np.ascontiguousarray(op.to_v2_layout(p4, N, K))).npu()
        x_dev = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).npu()

        # 1) v1（含 x 拆分）
        t1 = bench(lambda: op.run(p4_dev, s4t, x_dev, nE, N, K, lay=1))
        # 2) v2（自然 x，无拆分）
        t2 = bench(lambda: op.run(p4v2_dev, s4t, x_dev, nE, N, K, lay=2))
        # 3) 结果一致性（同一份参考）
        y1 = op.run(p4_dev, s4t, x_dev, nE, N, K, lay=1).cpu().numpy()
        y2 = op.run(p4v2_dev, s4t, x_dev, nE, N, K, lay=2).cpu().numpy()
        d = float(np.abs(y1 - y2).max())
        print(f"{name}:  v1 {t1:.3f} ms | v2 {t2:.3f} ms | 提速 {100*(t1-t2)/t1:+.1f}% | v1vs v2 maxerr {d:.2e}",
              flush=True)


if __name__ == "__main__":
    main()
