"""Single-token DeltaNet q/k normalization with original BF16 boundaries."""
import ctypes
import os
from pathlib import Path
import torch
from .norm_post import table

_lib = None

def run(values):
    if (values.shape != (1, 10240) or values.dtype != torch.bfloat16
            or values.device.type != 'npu' or not values.is_contiguous()):
        return None
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(os.environ.get('FNX_NORM_ORDERED_SO', str(Path(__file__).with_name('libfnx_norm_ordered.so'))))
        lib.fnx_norm_ordered.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3 + [ctypes.c_void_p]
        lib.fnx_norm_ordered.restype = ctypes.c_int
        _lib = lib
    output = torch.empty((1, 32, 128), dtype=values.dtype, device=values.device)
    rc = _lib.fnx_norm_ordered(values.data_ptr(), table(values.device).data_ptr(), output.data_ptr(),
                             32, 2048, 1, torch.npu.current_stream(values.device).npu_stream)
    if rc:
        raise RuntimeError(f'fused q/k normalization launch rejected: {rc}')
    return output[:, :16], output[:, 16:]
