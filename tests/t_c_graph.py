"""步骤 (c)：8 层串联 + NPUGraph 捕获，验证 graph-safe 与加速比。"""
import sys
import time

import torch

sys.path.insert(0, "/root/ascend-dev/fnx")
from fnx import engine as E                    # noqa: E402
from fnx.model import FnModel                  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 8
STEPS = 16


def bench(m, n=STEPS):
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        m.step(8309, with_head=False)
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    print(f"=== 步骤(c): {N} 层 graph-safe 验证 ===", flush=True)
    m = FnModel(P=2048, lpc=N, dev0="npu:0", dev1="npu:0", nlayers=N,
                with_head=False, verbose=True)

    # --- eager ---
    m.reset()
    for _ in range(3):
        m.step(8309, with_head=False)
    t_eager = bench(m)
    h_eager = m.h0.clone()

    # --- 捕获 ---
    print("\n=== 尝试 NPUGraph 捕获 (capture_error_mode='relaxed') ===", flush=True)
    ok = True
    try:
        m.capture()
    except Exception as ex:
        ok = False
        print("  >>> 捕获失败:", type(ex).__name__, str(ex)[:400])

    if not ok:
        return

    m.reset()
    for _ in range(3):
        m.step(8309, with_head=False)
    t_graph = bench(m)
    h_graph = m.h0.clone()
    d = (h_graph.float() - h_eager.float()).abs()
    rel = d.mean().item() / h_eager.float().abs().mean().clamp_min(1e-9).item()

    print(f"\n[eager] {t_eager:7.3f} ms/token → 单层 {t_eager/N:6.3f} ms")
    print(f"[graph] {t_graph:7.3f} ms/token → 单层 {t_graph/N:6.3f} ms  加速 {t_eager/t_graph:.2f}×")
    print(f"graph vs eager 输出: rel {rel*100:.4f}%  max_abs {d.max().item():.3e}")

    # --- 状态复位验证（文档 §7.8 的坑：连续 replay 不重置会发散） ---
    m.reset()
    out = []
    for _ in range(40):
        m.step(8309, with_head=False)
        out.append(float(m.h0.float().abs().max()))
    print(f"40 步 |h|max: 首 {out[0]:.3f} 末 {out[-1]:.3f}  NaN={any(x != x for x in out)}")


if __name__ == "__main__":
    main()
