"""Fused four-tap BF16 convolution and in-place history update."""
import ctypes
import os
from pathlib import Path

import torch

_lib = None
_tables = {}
_ROWS = 128


def _library():
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(os.environ.get("FNX_CONV_NATIVE_SO", str(Path(__file__).with_name("libfnx_conv.so"))))
        lib.fnx_conv_four.argtypes = [ctypes.c_void_p]*7 + [ctypes.c_int]*2 + [ctypes.c_void_p]
        lib.fnx_conv_four.restype = ctypes.c_int
        _lib = lib
    return _lib


def step(history, values, weights, order=1):
    dimension = values.numel()
    if (values.shape != (1, dimension) or history.shape != (1, dimension, 3)
            or weights.shape != (dimension, 4) or dimension % _ROWS):
        raise ValueError("unsupported convolution shape")
    if any(t.dtype != torch.bfloat16 or not t.is_contiguous() or t.device != values.device
           for t in (history, values, weights)):
        raise ValueError("convolution requires contiguous BF16 inputs on one device")
    key = str(values.device)
    if key not in _tables:
        packed = [2*(r*3+c if c<3 else _ROWS*3+r) for r in range(_ROWS) for c in range(4)]
        updated = [2*(r*3+c+1 if c<2 else _ROWS*3+r) for r in range(_ROWS) for c in range(3)]
        columns = [r*16 for r in range(_ROWS)]
        _tables[key] = tuple(torch.tensor(index, dtype=torch.int32, device=values.device)
                             for index in (packed, updated, columns))
    output = torch.empty_like(values)
    rc = _library().fnx_conv_four(history.data_ptr(), values.data_ptr(), weights.data_ptr(),
                                  output.data_ptr(), *(t.data_ptr() for t in _tables[key]),
                                  dimension, order, torch.npu.current_stream(values.device).npu_stream)
    if rc:
        raise RuntimeError(f"convolution launch rejected: {rc}")
    return output
