"""★ 量化误差分解：到底是谁把专家权重搞坏的？

背景：`load.py` 在 `FNX_EXP_G=64` 时把 g32 fp16 的 scale 压成 "g64 + int8 + 每行 fp16 base"，
文档里记的代价是"专家量化误差 10.02% → 约 12%"。这个量级对 W4 g32 来说太大（正常 1–2%），
所以必须拆开看是三件事里的哪一件：

  A. 组从 32 变 64（同一组内 64 个 k 共用一个 scale）
  B. 4bit 权重在 g64 下重新取整（`merge_scales_g64` 里的 round/clamp）
  C. scale 从 fp16 存成 int8（+ 每行 fp16 base）

本脚本纯 CPU/host，不需要 NPU，也不占用服务器上的模型。

用法（容器内）：
  cd /root/ascend-dev/fnx && /root/ascend-dev/venv/bin/python tools/qerr_ab.py
"""
import os

import numpy as np
import torch
from safetensors.torch import load_file

from fnx import engine as E
from fnx.load import QDIR

G32 = 32


def deq(q, s, gs):
    """q: int8 [-8..7] [n,N,K] 容器   s: [n,N,K/gs] -> fp32 [n,N,K]"""
    n, N, K = q.shape
    w = q.float().view(n, N, K // gs, gs) * s.float().unsqueeze(-1)
    return w.reshape(n, N, K)


def rel(a, b):
    """a vs 参考 b 的相对误差（均值口径）+ 最大绝对误差"""
    d = (a - b).abs()
    return float(d.mean() / (b.abs().mean() + 1e-12)), float(d.max())


def one(tag, q4, s4, nE):
    """q4: [E,N,K] int8 容器, s4: [E,N,K/32] fp16"""
    q4 = q4[:nE]
    s4 = s4[:nE]
    n, N, K = q4.shape
    gs_in = K // s4.shape[-1]
    assert gs_in == G32, f"期望 g32，实际 gs_in={gs_in}"

    ref = deq(q4, s4, gs_in)                       # g32 fp16 = 真值基准

    # 变体 1：g64 + fp16 scale（只改组大小 + 重新取整）
    q64, s64, _ = E.merge_scales_g64(q4, s4, gs_in=gs_in, gs_out=64, to_int8=False)
    v1 = deq(q64, s64, 64)

    # 变体 2：g64 + int8 scale + 每行 fp16 base（= 线上 FNX_EXP_G=64）
    q64i, s8, base = E.merge_scales_g64(q4, s4, gs_in=gs_in, gs_out=64)
    v2 = deq(q64i, s8.float() * base.float().unsqueeze(-1), 64)

    r1 = rel(v1, ref)
    r2 = rel(v2, ref)
    r12 = rel(v2, v1)
    print(f"  {tag:14s} N={N} K={K}")
    print(f"    g32 fp16  (基准)                     |W|mean={ref.abs().mean():.5f}")
    print(f"    g64 fp16   vs g32fp16 : rel {r1[0]*100:7.3f}%  max {r1[1]:.5f}")
    print(f"    g64 i8+base vs g32fp16: rel {r2[0]*100:7.3f}%  max {r2[1]:.5f}")
    print(f"    g64 i8+base vs g64fp16: rel {r12[0]*100:7.3f}%  max {r12[1]:.5f}"
          f"      <- 这一项才是「int8 存 scale」的代价", flush=True)
    return r1[0], r2[0], r12[0]


def main():
    print(f"QDIR = {QDIR}")
    layers = [0, 10, 45]
    nE = 4
    rows = []
    for li in layers:
        p = os.path.join(QDIR, f"L{li:02d}.safetensors")
        if not os.path.exists(p):
            print(f"  跳过 L{li:02d}（不存在）")
            continue
        sd = load_file(p)
        print(f"\n=== L{li:02d} ===")
        for want in ("gate_up_proj", "down_proj"):
            kq = [k for k in sd if want in k and k.endswith(".q4")]
            if not kq:
                print(f"  {want}: 找不到 q4 键")
                continue
            kq, ks = kq[0], [k for k in sd if want in k and k.endswith(".s4")][0]
            r = one(f"L{li:02d} {want.replace('_proj','')}", sd[kq], sd[ks], nE)
            rows.append((li, want, r))
        del sd

    print("\n================ 汇总（rel_mean） ================")
    print(f"{'layer':>6} {'proj':>10} {'g64fp16':>9} {'g64i8':>9} {'i8 代价':>9}")
    for li, w, (a, b, c) in rows:
        print(f"{li:>6} {w.replace('_proj',''):>10} {a*100:>8.3f}% {b*100:>8.3f}% {c*100:>8.3f}%")
    if rows:
        a = np.mean([r[2][0] for r in rows])
        b = np.mean([r[2][1] for r in rows])
        c = np.mean([r[2][2] for r in rows])
        print(f"{'均值':>6} {'':>10} {a*100:>8.3f}% {b*100:>8.3f}% {c*100:>8.3f}%")


if __name__ == "__main__":
    main()
