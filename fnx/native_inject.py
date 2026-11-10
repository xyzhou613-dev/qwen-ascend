"""Decode residual injection preserving the intermediate BF16 rounding."""
import ctypes
import os
from pathlib import Path
import torch

_lib = None


def inject(hyper, gate, values):
    if (hyper.shape != (1, 1, 10240) or values.numel() != 2560
            or gate.numel() != 4):
        return None
    if any(t.dtype != torch.bfloat16 or not t.is_contiguous() or t.device != hyper.device
           for t in (hyper, gate, values)) or hyper.device.type != 'npu':
        return None
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(os.environ.get('FNX_INJECT_SO', str(Path(__file__).with_name('libfnx_inject.so'))))
        lib.fnx_inject.argtypes = [ctypes.c_void_p]*4 + [ctypes.c_int, ctypes.c_void_p]
        lib.fnx_inject.restype = ctypes.c_int
        _lib = lib
    output = torch.empty_like(hyper)
    rc = _lib.fnx_inject(hyper.data_ptr(), values.data_ptr(), gate.data_ptr(), output.data_ptr(), 1,
                         torch.npu.current_stream(hyper.device).npu_stream)
    if rc:
        raise RuntimeError(f'residual injection launch rejected: {rc}')
    return output
