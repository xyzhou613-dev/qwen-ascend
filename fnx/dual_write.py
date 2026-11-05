"""Exact BF16 dual-cache writes for contiguous token positions."""
import ctypes
import os
from pathlib import Path

import torch

_lib = None


def _library():
    global _lib
    if _lib is None:
        path = os.environ.get("FNX_DUAL_WRITE_SO", str(Path(__file__).with_name("libfnx_dual_write.so")))
        lib = ctypes.CDLL(path)
        lib.fnx_write_dual_kv.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 4 + [ctypes.c_void_p]
        lib.fnx_write_dual_kv.restype = None
        _lib = lib
    return _lib


def write(store, index, k, v, pos):
    """Write both layouts in place. The caller constructs pos as start + arange(T)."""
    tokens = k.shape[0]
    for tensor in (k, v):
        if (tensor.shape != (tokens, 2, 256) or tensor.dtype != torch.bfloat16
                or tensor.stride()[1:] != (256, 1) or tensor.stride(0) < 512
                or tensor.stride(0) % 16 or tensor.stride(0) // 16 >= 65536):
            return False
    if pos.dtype != torch.int64 or pos.numel() != tokens or not 0 < tokens <= store.P:
        return False
    targets = (store.k[index], store.v[index], store.k[index + store.n], store.v[index + store.n])
    _library().fnx_write_dual_kv(
        k.data_ptr(), v.data_ptr(), pos.data_ptr(), *(t.data_ptr() for t in targets),
        tokens, store.P, k.stride(0), v.stride(0),
        torch.npu.current_stream(k.device).npu_stream,
    )
    return True
