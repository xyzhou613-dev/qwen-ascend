#!/usr/bin/env python3
"""比对 AscendC W4A16 GEMV 输出与黄金数据。

用法：
  python tools/cmp_w4gemv.py gate_up_proj /path/to/kernel_out.bin
  python tools/cmp_w4gemv.py --selftest          # 用 torch 重算一遍，验证黄金数据自洽
输出：rel_mean / rel_max，按 meta.json 的容差判定 PASS/FAIL。
"""
import json
import os
import sys

import numpy as np

GOLD = "/root/ascend-dev/golden/w4gemv"


def load_bf16(path):
    return np.fromfile(path, dtype=np.int16).view(np.uint16).astype(np.uint32) << 16


def bf16_to_f32(raw_i16):
    """int16 承载的 bf16 位型 → float32"""
    return (raw_i16.astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bits(x):
    return (x.astype(np.uint32) >> 16).astype(np.uint16)


def deq_reference(which, meta):
    """纯 numpy 按契约反量化，作为独立于 torch 的第二道校验。"""
    c = meta["cases"][which]
    N, K, gs = c["N"], c["K"], c["gs"]
    p4 = np.fromfile(os.path.join(GOLD, c["p4"]["file"]), dtype=np.int8).reshape(N, K // 2)
    s4 = np.fromfile(os.path.join(GOLD, c["s4"]["file"]), dtype=np.float16).reshape(N, K // gs)
    x = bf16_to_f32(np.fromfile(os.path.join(GOLD, c["x"]["file"]), dtype=np.int16)).reshape(K)
    b = p4.view(np.uint8)
    lo = ((b & 0x0F).astype(np.int8) ^ 8) - 8
    hi = (((b >> 4) & 0x0F).astype(np.int8) ^ 8) - 8
    w4 = np.empty((N, K), dtype=np.int8)
    w4[:, 0::2] = lo
    w4[:, 1::2] = hi
    w = w4.astype(np.float32) * np.repeat(s4.astype(np.float32), gs, axis=1)
    return (w @ x).astype(np.float32)


def cmp(which, out_path):
    meta = json.load(open(os.path.join(GOLD, "meta.json")))
    c = meta["cases"][which]
    ref = bf16_to_f32(np.fromfile(os.path.join(GOLD, c["y_ref"]["file"]), dtype=np.int16))
    got = np.fromfile(out_path, dtype=np.float32) if os.path.getsize(out_path) == c["N"] * 4 \
        else bf16_to_f32(np.fromfile(out_path, dtype=np.int16))
    assert got.shape == ref.shape, (got.shape, ref.shape)

    # 主判据：与「numpy 按契约独立重算」的 fp32 参考比。
    # 原因（第 26 轮实测）：y_ref 存成 bf16，它自己相对 numpy 契约就有 0.256% 的固有误差，
    # 拿 kernel 去比 y_ref 会把 bf16 的存储损失算到 kernel 头上（旧版因此每轮误判 FAIL）。
    vref = deq_reference(which, meta)

    def rel(a, b):
        d = np.abs(a - b)
        scale = np.abs(b).mean() + 1e-12
        return float(d.mean() / scale), float(d.max() / scale)

    rel_mean, rel_max = rel(got, vref)
    rel_mean_bf, rel_max_bf = rel(got, ref)
    tol = meta["tolerance"]
    ok = rel_mean <= tol["rel_mean"] and rel_max <= tol["rel_max"]
    print(f"{which:14s} N={c['N']} K={c['K']}")
    print(f"  vs numpy 契约参考(fp32) : rel_mean {rel_mean*100:8.4f}%  rel_max {rel_max*100:8.4f}%  "
          f"-> {'PASS' if ok else 'FAIL'}   (门限 {tol['rel_mean']*100:.2f}% / {tol['rel_max']*100:.2f}%)")
    print(f"  vs y_ref(bf16，含存储损失) : rel_mean {rel_mean_bf*100:8.4f}%  rel_max {rel_max_bf*100:8.4f}%  (仅供参考)")
    return ok


def deq_multi(base, meta, nE):
    """10 专家批次的 numpy 契约参考（独立实现，不依赖 C++/torch）。"""
    c = meta["cases"]["multi_" + base]
    N, K, gs = c["N"], c["K"], c["gs"]
    d = os.path.join(GOLD, c["dir"])
    p4 = np.fromfile(os.path.join(d, base + "_p4.bin"), dtype=np.int8).reshape(nE, N, K // 2)
    s4 = np.fromfile(os.path.join(d, base + "_s4.bin"), dtype=np.float16).reshape(nE, N, K // gs)
    x = bf16_to_f32(np.fromfile(os.path.join(d, base + "_x.bin"), dtype=np.int16)).reshape(nE, K)
    b = p4.view(np.uint8)
    lo = ((b & 0x0F).astype(np.int8) ^ 8) - 8
    hi = (((b >> 4) & 0x0F).astype(np.int8) ^ 8) - 8
    w4 = np.empty((nE, N, K), dtype=np.int8)
    w4[:, :, 0::2] = lo
    w4[:, :, 1::2] = hi
    w = w4.astype(np.float32) * np.repeat(s4.astype(np.float32), gs, axis=2)
    return np.einsum("enk,ek->en", w, x).astype(np.float32), c


def cmp_multi(which, out_path):
    """grouped（10 专家一次 launch）的验收：主判据同样是 numpy 契约参考。"""
    meta = json.load(open(os.path.join(GOLD, "meta.json")))
    base = which.replace("multi_", "")
    c = meta["cases"][which]
    nE = c["B"]
    vref, _ = deq_multi(base, meta, nE)
    got = np.fromfile(out_path, dtype=np.float32)
    assert got.size == nE * c["N"], (got.size, nE * c["N"])
    got = got.reshape(nE, c["N"])
    d = np.abs(got - vref)
    scale = np.abs(vref).mean() + 1e-12
    rel_mean, rel_max = float(d.mean() / scale), float(d.max() / scale)
    tol = meta["tolerance"]
    ok = rel_mean <= tol["rel_mean"] and rel_max <= tol["rel_max"]
    print(f"{which:16s} B={nE} N={c['N']} K={c['K']}")
    print(f"  vs numpy 契约参考(fp32) : rel_mean {rel_mean*100:8.4f}%  rel_max {rel_max*100:8.4f}%  "
          f"-> {'PASS' if ok else 'FAIL'}   (门限 {tol['rel_mean']*100:.2f}% / {tol['rel_max']*100:.2f}%)")
    return ok


def selftest():
    meta = json.load(open(os.path.join(GOLD, "meta.json")))
    allok = True
    for which in ("gate_up_proj", "down_proj"):
        ref = bf16_to_f32(np.fromfile(
            os.path.join(GOLD, meta["cases"][which]["y_ref"]["file"]), dtype=np.int16))
        mine = deq_reference(which, meta)
        d = np.abs(mine - ref)
        rel = float(d.mean() / (np.abs(ref).mean() + 1e-12))
        # 期望差异仅来自 bf16 舍入（y_ref 存成 bf16）
        ok = rel <= 5e-3
        allok &= ok
        print(f"  selftest {which:14s} numpy 契约重算 vs y_ref(bf16): rel {rel*100:7.4f}%  "
              f"-> {'OK' if ok else 'MISMATCH'}")
    print("黄金数据自洽:", "PASS" if allok else "FAIL —— 契约或数据有问题，先修这个再写 kernel")
    return allok


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) >= 3 and sys.argv[1].startswith("multi_"):
        sys.exit(0 if cmp_multi(sys.argv[1], sys.argv[2]) else 1)
    elif len(sys.argv) >= 3:
        sys.exit(0 if cmp(sys.argv[1], sys.argv[2]) else 1)
    else:
        print(__doc__)
