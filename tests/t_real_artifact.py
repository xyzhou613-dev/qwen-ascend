"""用真实 QP 产物（/models/Qwen3.8-Flash-Next-QP/L00.packed.safetensors）验证接入层。

目的（golden 数据证明不了的部分）：
  1) 真产物里的张量键名 / 形状 / dtype 与接入层假设一致
  2) 装配期 s4 转置能在真实 [512][N][K/gs] 上跑
  3) 10 个真实专家的 p4 能被 kernel 正确消费（用 numpy 契约独立对拍）

用法：python tests/t_real_artifact.py [layer=L00] [nE=10]
"""
import json
import os
import sys

import numpy as np
import torch

from fnx.w4gemv_op import W4GemvOp, GS

QP = "/models/Qwen3.8-Flash-Next-QP"


def find_key(f, cands, pred):
    keys = list(f.keys())
    for c in cands:
        for k in keys:
            if c in k and pred(k):
                return k
    return None


def main():
    layer = sys.argv[1] if len(sys.argv) > 1 else "L00"
    nE = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    path = os.path.join(QP, layer + ".packed.safetensors")
    if not os.path.exists(path):
        print(f"[real] 找不到 {path}")
        return 1
    print(f"[real] {path}  {os.path.getsize(path)/2**30:.2f} GiB", flush=True)

    try:
        from safetensors import safe_open
    except ImportError:
        print("[real] 容器内没有 safetensors")
        return 1

    op = W4GemvOp()
    with safe_open(path, framework="pt") as f:
        keys = list(f.keys())
        print(f"[real] 张量数 {len(keys)}；前 8 个: {keys[:8]}", flush=True)
        for want in ("gate_up_proj", "down_proj"):
            kp = find_key(f, [want], lambda k: k.endswith(".p4") or ".p4" in k)
            ks = find_key(f, [want], lambda k: k.endswith(".s4") or ".s4" in k)
            print(f"[real] {want}: p4={kp}  s4={ks}", flush=True)
            if kp is None or ks is None:
                print(f"[real] !! {want} 键名没匹配上，跳过（把上面的 keys 列表拿去对照）")
                continue

            p4_all = f.get_slice(kp)
            s4_all = f.get_slice(ks)
            shp, sshp = p4_all.get_shape(), s4_all.get_shape()
            print(f"[real]   p4 shape={shp} dtype={p4_all.get_dtype()}  s4 shape={sshp}", flush=True)
            E, N, Kh = shp[0], shp[1], shp[2] * 2          # Kh = K
            p4 = p4_all[0:nE].numpy() if hasattr(p4_all[0:nE], "numpy") else np.asarray(p4_all[0:nE])
            s4 = s4_all[0:nE].numpy() if hasattr(s4_all[0:nE], "numpy") else np.asarray(s4_all[0:nE])
            print(f"[real]   取前 {nE} 个专家: p4 {p4.shape} {p4.dtype}, s4 {s4.shape} {s4.dtype}", flush=True)

            rng = np.random.RandomState(0)
            x = (rng.randn(nE, Kh).astype(np.float32) * 0.3)

            s4t_dev = op.prepare_layer(s4.astype(np.float16))          # 装配期
            p4_dev = torch.from_numpy(np.ascontiguousarray(p4, dtype=np.int8)).npu()
            y = op.run(p4_dev, s4t_dev, x, nE, N, Kh).cpu().numpy()     # decode

            b = p4.view(np.uint8)
            lo = ((b & 0x0F).astype(np.int8) ^ 8) - 8
            hi = (((b >> 4) & 0x0F).astype(np.int8) ^ 8) - 8
            w4 = np.empty((nE, N, Kh), dtype=np.int8)
            w4[:, :, 0::2] = lo
            w4[:, :, 1::2] = hi
            ref = np.einsum("enk,ek->en", w4.astype(np.float32) *
                            np.repeat(s4.astype(np.float32), GS, axis=2), x)
            d = np.abs(y - ref)
            rel = float(d.mean() / (np.abs(ref).mean() + 1e-12))
            ok = rel <= 1e-3
            print(f"[real]   {want}: N={N} K={Kh}  rel_mean {rel*100:.4f}%  超差 {int((d > 1e-3*np.abs(ref)+1e-6).sum())}/{d.size}"
                  f"  -> {'PASS' if ok else 'FAIL'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
