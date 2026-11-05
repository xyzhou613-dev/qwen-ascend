"""Fuse FP32 indexer group scoring and causal masking for aligned tiles."""
import ctypes
import os
from pathlib import Path

import torch

_lib = None


def score(input, pos):
    if (input.dtype != torch.float32 or input.ndim != 2
            or not input.is_contiguous() or input.shape[0] != 4 * pos.numel()
            or input.shape[1] % 2048 or not pos.is_contiguous() or pos.dtype != torch.int64):
        return None
    global _lib
    if _lib is None:
        path = os.environ.get("FNX_SCORE_SO", str(Path(__file__).with_name("libfnx_score.so")))
        lib = ctypes.CDLL(path)
        lib.fnx_score_groups.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 2 + [ctypes.c_void_p]
        lib.fnx_score_groups.restype = None
        _lib = lib
    tokens, groups = pos.numel(), input.shape[1]
    output = torch.empty((tokens, groups), dtype=input.dtype, device=input.device)
    _lib.fnx_score_groups(input.data_ptr(), pos.data_ptr(), output.data_ptr(),
                          tokens, groups, torch.npu.current_stream(input.device).npu_stream)
    return output
