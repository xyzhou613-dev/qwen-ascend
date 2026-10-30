"""kernel 从 Python 调用（ctypes 桥）验收：与 numpy 契约参考对拍 + 测时。

用法（容器内）：
  /root/ascend-dev/venv/bin/python tests/t_lib_w4gemv.py multi_gate_up
依赖：/root/ascend-dev/ascendc/direct/libfnxw4gemv.so
"""
import ctypes
import json
import os
import sys
import time

import numpy as np
import torch
import torch_npu  # noqa: F401

GOLD = "/root/ascend-dev/golden/w4gemv"
LIB = "/root/ascend-dev/ascendc/direct/libfnxw4gemv.so"
GS = 32


def bf16_to_f32(raw_i16):
    return (raw_i16.astype(np.uint32) << 16).view(np.float32)


def build(name, nE):
    """读 golden multi 数据，按 kernel 要的布局准备好 xe/xo/s4T。"""
    meta = json.load(open(os.path.join(GOLD, "meta.json")))
    base = name.replace("multi_", "")
    c = meta["cases"][name]
    N, K = c["N"], c["K"]
    d = os.path.join(GOLD, c["dir"])
    p4 = np.fromfile(os.path.join(d, base + "_p4.bin"), dtype=np.int8).reshape(nE, N, K // 2)
    s4 = np.fromfile(os.path.join(d, base + "_s4.bin"), dtype=np.float16).reshape(nE, N, K // GS)
    x = bf16_to_f32(np.fromfile(os.path.join(d, base + "_x.bin"), dtype=np.int16)).reshape(nE, K)

    # 偶/奇 lane 拆分
    xe = np.ascontiguousarray(x[:, 0::2], dtype=np.float32)
    xo = np.ascontiguousarray(x[:, 1::2], dtype=np.float32)
    # scale 转置成 [nE][K/gs][N]
    s4t = np.ascontiguousarray(np.transpose(s4, (0, 2, 1)), dtype=np.float16)

    # numpy 契约参考
    b = p4.view(np.uint8)
    lo = ((b & 0x0F).astype(np.int8) ^ 8) - 8
    hi = (((b >> 4) & 0x0F).astype(np.int8) ^ 8) - 8
    w4 = np.empty((nE, N, K), dtype=np.int8)
    w4[:, :, 0::2] = lo
    w4[:, :, 1::2] = hi
    ref = np.einsum("enk,ek->en",
                    w4.astype(np.float32) * np.repeat(s4.astype(np.float32), GS, axis=2), x)
    return p4, s4t, xe, xo, ref.astype(np.float32), N, K


def main():
    case = sys.argv[1] if len(sys.argv) > 1 else "multi_gate_up_proj"
    nblk = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    nE = 10

    lib = ctypes.CDLL(LIB)
    lib.fnx_w4gemv_launch.restype = ctypes.c_int
    lib.fnx_w4gemv_launch.argtypes = [ctypes.c_void_p] * 6 + [ctypes.c_int] * 4 + [ctypes.c_void_p]

    p4, s4t, xe, xo, ref, N, K = build(case, nE)
    print(f"[lib] case={case} N={N} K={K} nE={nE} nblk={nblk}", flush=True)

    def to_dev(a):
        return torch.from_numpy(a).npu()

    dp4, ds4t, dxe, dxo = to_dev(p4), to_dev(s4t), to_dev(xe), to_dev(xo)
    dy = torch.zeros((nE, N), dtype=torch.float32).npu()
    ddbg = torch.zeros(65536, dtype=torch.float32).npu()

    st = 0
    try:
        st = int(torch.npu.current_stream().npu_stream)
    except Exception:  # noqa: BLE001
        st = 0
    print(f"[lib] stream=0x{st:x}" if st else "[lib] stream=默认", flush=True)

    args = (ctypes.c_void_p(dp4.data_ptr()), ctypes.c_void_p(ds4t.data_ptr()),
            ctypes.c_void_p(dxe.data_ptr()), ctypes.c_void_p(dxo.data_ptr()),
            ctypes.c_void_p(dy.data_ptr()), ctypes.c_void_p(ddbg.data_ptr()),
            N, K, nE, nblk, ctypes.c_void_p(st))

    # warmup + 计时
    for r in range(3):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        rc = lib.fnx_w4gemv_launch(*args)
        torch.npu.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        if rc != 0:
            print(f"[lib] FATAL rc={rc}")
            return 1
        print(f"[lib] rep{r} python->kernel = {ms:.3f} ms", flush=True)

    got = dy.cpu().numpy()
    d = np.abs(got - ref)
    scale = np.abs(ref).mean() + 1e-12
    rel_mean, rel_max = float(d.mean() / scale), float(d.max() / scale)
    tol = json.load(open(os.path.join(GOLD, "meta.json")))["tolerance"]
    ok = rel_mean <= tol["rel_mean"] and rel_max <= tol["rel_max"]
    print(f"[lib] vs numpy 契约: rel_mean {rel_mean*100:.4f}%  rel_max {rel_max*100:.4f}%  "
          f"-> {'PASS' if ok else 'FAIL'}（门限 {tol['rel_mean']*100:.2f}%/{tol['rel_max']*100:.2f}%）", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
