#!/usr/bin/env python3
"""把 Q 目录的专家权重（int8 容器，1 值/字节）重打包为**真 4-bit**（2 值/字节）落盘。

为什么需要：
  `quantize_w4.py` 产出的 `.q4` 是 int8 容器，一个 4-bit 值占 1 字节 →
  "W4 专家"实测 2.344 GiB/层，48 层 112.5 GiB，2×60.5 GiB 装不下。
  真 4-bit 打包后 1.172 GiB/层 → 每卡 24 层 ≈ 33.4 GiB，余 26 GiB 给 KV/workspace。

打包布局（供 kernel 读取，勿改）：
  对最后一维（K）成对打包：byte j 的低 4 bit = W[k=2j]，高 4 bit = W[k=2j+1]
  4-bit 值以二进制补码存放在 0..15：value = ((nibble ^ 8) - 8)，范围 -8..7
  scale 原样搬运（fp16，per-group-32），dequant: W[n,k] = nibble(n,k) * s[n, k//32]

用法：
  python tools/repack_q4.py verify          # 只校验（第 0/1/24/47 层）
  python tools/repack_q4.py run [lo] [hi]   # 全量或区间，可断点续跑
"""
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

SRC = "/models/Qwen3.8-Flash-Next-Q"
OUT = "/models/Qwen3.8-Flash-Next-QP"
NLAYER = 48
EXPERT_KEYS = ("gate_up_proj", "down_proj")
CHUNK_E = 32          # 按专家分块，把 numpy 临时量控制在 ~200 MB


def _key(layer, which, suf):
    return f"model.language_model.layers.{layer}.mlp.experts.{which}.{suf}"


def pack_np(q):
    """q: np.int8 [E,N,K] (C 连续) → np.uint8 [E,N,K/2]，无损。"""
    E, N, K = q.shape
    assert K % 4 == 0, (K, "最后一维必须能被 4 整除（int32 视图）")
    out = np.empty((E, N, K // 2), dtype=np.uint8)
    for e0 in range(0, E, CHUNK_E):
        e1 = min(e0 + CHUNK_E, E)
        a = q[e0:e1].view(np.int32)                       # 一个 int32 = 相邻 4 个 int8
        v = ((a & 0x0F)
             | (((a >> 8) & 0x0F) << 4)
             | (((a >> 16) & 0x0F) << 8)
             | (((a >> 24) & 0x0F) << 12))               # 2 个打包字节装进一个 int16
        out[e0:e1] = v.astype(np.uint16).view(np.uint8).reshape(e1 - e0, N, K // 2)
    return out


def unpack_np(p):
    """参考解包（仅供校验；运行时由 kernel 完成）。np.uint8 [E,N,K/2] → np.int8 [E,N,K]"""
    E, N, K2 = p.shape
    out = np.empty((E, N, K2 * 2), dtype=np.int8)
    lo = ((p & 0x0F).astype(np.int8) ^ 8) - 8
    hi = (((p >> 4) & 0x0F).astype(np.int8) ^ 8) - 8
    out[..., 0::2] = lo
    out[..., 1::2] = hi
    return out


def do_layer(i, verify=False):
    fp = os.path.join(OUT, f"L{i:02d}.packed.safetensors")
    if os.path.exists(fp) and not verify:
        return None
    payload = {}
    info = {}
    with safe_open(os.path.join(SRC, f"L{i:02d}.safetensors"), framework="np", device="cpu") as f:
        for which in EXPERT_KEYS:
            kq, ks = _key(i, which, "q4"), _key(i, which, "s4")
            q = np.ascontiguousarray(f.get_tensor(kq))
            s = f.get_tensor(ks)
            assert q.dtype == np.int8 and s.dtype == np.float16
            t0 = time.time()
            p = pack_np(q)
            dt = time.time() - t0
            if verify:
                qb = unpack_np(p)
                if not np.array_equal(qb, q):
                    bad = np.argwhere(qb != q)[:5]
                    raise SystemExit(f"[L{i:02d}] {which} 打包**非无损**！首个不一致 {bad.tolist()}")
            payload[_key(i, which, "p4")] = torch.from_numpy(p)
            payload[_key(i, which, "s4")] = torch.from_numpy(s.copy())
            info[which] = {"shape": list(q.shape), "scale_shape": list(s.shape),
                           "gs": q.shape[-1] // s.shape[-1],
                           "q4_bytes": int(q.nbytes), "p4_bytes": int(p.nbytes),
                           "s4_bytes": int(s.nbytes), "pack_secs": round(dt, 2)}
            del q, p
    if not verify:
        os.makedirs(OUT, exist_ok=True)
        save_file(payload, fp)
        info["file_bytes"] = os.path.getsize(fp)
        return info
    return info


def verify(layers=(0, 1, 24, 47)):
    ok = True
    for i in layers:
        info = do_layer(i, verify=True)
        print(f"[L{i:02d}] 无损校验通过  " + "  ".join(
            f"{k}: {tuple(v['shape'])} gs={v['gs']} {v['q4_bytes']/2**20:.0f}MiB→{v['p4_bytes']/2**20:.0f}MiB "
            f"({v['pack_secs']}s)" for k, v in info.items()), flush=True)
    return ok


def run(lo, hi):
    os.makedirs(OUT, exist_ok=True)
    man_path = os.path.join(OUT, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {
        "format": "fnx-w4-packed-v1",
        "source": SRC,
        "layout": {
            "pack_axis": "last (K)",
            "byte_j": {"low_nibble": "k=2j", "high_nibble": "k=2j+1"},
            "nibble_encoding": "two's complement in 0..15, value = ((nibble ^ 8) - 8), range -8..7",
            "dequant": "W[n,k] = nibble(n,k) * s4[n, k // gs]",
            "tensor_names": {"packed": "<key>.p4", "scale": "<key>.s4"},
        },
        "layers": {},
    }
    t0 = time.time()
    for i in range(lo, hi):
        r = do_layer(i)
        if r is None:
            continue
        man["layers"][f"L{i:02d}"] = r
        json.dump(man, open(man_path, "w"), indent=1)
        tot = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".safetensors"))
        per = sum(r[w]["p4_bytes"] + r[w]["s4_bytes"] for w in EXPERT_KEYS)
        print(f"[L{i:02d}] {per/2**30:.3f} GiB 打包完成  累计 {tot/2**30:.1f} GiB  "
              f"{(time.time()-t0):.0f}s", flush=True)
    tot = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT) if f.endswith(".safetensors"))
    print(f"=== 输出累计 {tot/2**30:.1f} GiB，{len(man['layers'])} 层 ===", flush=True)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "verify"
    if cmd == "verify":
        verify()
    elif cmd == "run":
        lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        hi = int(sys.argv[3]) if len(sys.argv) > 3 else NLAYER
        run(lo, hi)
    else:
        print(__doc__)
