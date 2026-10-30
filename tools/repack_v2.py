"""★ 离线生成 LQP artifact：v2 打包 + **offset-binary** 重排 + 尺度。

解决两个问题：
1. 冷启动 818 s 里有约 363 s 花在运行时做 v2 打包（7.9 s/层）→ 落盘后直接读；
2. 运行时解包原本要 6 个算子还原符号 `(n^8)-8`；改成 **offset-binary**（n' = value+8）后
   只需 `n' - 8` 一个减法 → 每层解包从 +0.88 ms 降到预期 +0.5 ms。

产物 `/models/Qwen3.8-Flash-Next-LQP/L{nn}.safetensors`，每层含（键名用短名）：
    <proj>.v2o   uint8 [E, N, K/2]   offset-binary v2 打包（byte(64s+j): lo=W[128s+j]+8, hi=W[128s+64+j]+8）
    <proj>.s8    int8  [E, N, K/64]  g64 缩放（与 merge_scales_g64 的 to_int8 输出一致）
    <proj>.base  fp16  [E, N]        每行基准

用法（容器内，纯 CPU，可后台跑）：
  cd /root/ascend-dev/fnx && python tools/repack_v2.py [--layers 0-47]
已存在的层会跳过，可续跑。
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from fnx import engine as E
from fnx.load import QDIR

OUT = os.environ.get("FNX_LQP", "/models/Qwen3.8-Flash-Next-LQP")
# ★ scale 组大小：64（基线，37.5 MiB/层）或 128（减半，18.8 MiB/层）。
#   第 32 轮实测：g128 的 mean_nll 与 g64 差别在噪声内（2.1657 vs 2.1756），
#   省下的 18.8 MiB/层 × 45 层 ≈ 846 MiB 可再预反量化 ~15 层注意力投影（约 −11 ms）。
GS_OUT = int(os.environ.get("FNX_LQP_GS", "64"))
PROJS = ("gate_up_proj", "down_proj")
CHUNK = 64


def pack_v2_ob(q):
    """int8 容器 [E,N,K]（值 -8..7）-> offset-binary v2 打包 uint8 [E,N,K/2]（分块处理）。"""
    E_, N, K = q.shape
    S = K // 128
    out = np.empty((E_, N, K // 2), dtype=np.uint8)
    for e0 in range(0, E_, CHUNK):
        e1 = min(e0 + CHUNK, E_)
        v = q[e0:e1].reshape(e1 - e0, N, S, 2, 64).astype(np.int16) + 8      # 0..15
        if v.min() < 0 or v.max() > 15:
            raise ValueError(f"越界: min={v.min()} max={v.max()}")
        lo = v[:, :, :, 0, :].astype(np.uint8)
        hi = v[:, :, :, 1, :].astype(np.uint8)
        o = np.empty((e1 - e0, N, S, 64), dtype=np.uint8)
        o[...] = lo | (hi << 4)
        out[e0:e1] = o.reshape(e1 - e0, N, K // 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0-47")
    a = ap.parse_args()
    lo_, hi_ = (int(x) for x in a.layers.split("-"))
    os.makedirs(OUT, exist_ok=True)
    print(f"源 {QDIR}\n产物 {OUT}\n层 {lo_}..{hi_}", flush=True)

    t_all = time.time()
    for i in range(lo_, hi_ + 1):
        dst = os.path.join(OUT, f"L{i:02d}.safetensors")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"  L{i:02d} 已存在，跳过", flush=True)
            continue
        t0 = time.time()
        src = os.path.join(QDIR, f"L{i:02d}.safetensors")
        with safe_open(src, framework="pt", device="cpu") as f:
            keys = list(f.keys())
            sd = {}
            for proj in PROJS:
                kq = [k for k in keys if f".{proj}.q4" in k or k.endswith(f"{proj}.q4")]
                ks = [k for k in keys if f".{proj}.s4" in k or k.endswith(f"{proj}.s4")]
                if not kq or not ks:
                    raise RuntimeError(f"L{i:02d} 找不到 {proj} 的 q4/s4")
                q4 = f.get_tensor(kq[0])
                s4 = f.get_tensor(ks[0])
                gs_in = q4.shape[-1] // s4.shape[-1]
                q2, s8, base = E.merge_scales_g64(q4, s4, gs_in=gs_in, gs_out=GS_OUT)
                packed = pack_v2_ob(q2.numpy())
                del q2
                sd[f"mlp.experts.{proj}.v2o"] = torch.from_numpy(packed)
                sd[f"mlp.experts.{proj}.s8"] = s8.clone()
                sd[f"mlp.experts.{proj}.base"] = base.clone()
                print(f"    {proj}: E={packed.shape[0]} N={packed.shape[1]} "
                      f"K/2={packed.shape[2]} 打包 {packed.nbytes/2**20:.0f} MiB", flush=True)
                del packed
        # 先写临时文件再改名，避免中断留下半个文件
        tmp = dst + ".tmp"
        save_file(sd, tmp)
        os.replace(tmp, dst)
        mb = os.path.getsize(dst) / 2**20
        print(f"  L{i:02d} 完成 {mb:.0f} MiB  {time.time()-t0:.1f}s "
              f"（累计 {time.time()-t_all:.0f}s）", flush=True)
        del sd

    print(f"\n全部完成，用时 {time.time()-t_all:.0f}s", flush=True)
    json.dump({"layers": f"{lo_}-{hi_}", "layout": "v2-offset-binary", "gs": GS_OUT,
               "keys": [f"mlp.experts.{p}.{x}" for p in PROJS for x in ("v2o", "s8", "base")]},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
