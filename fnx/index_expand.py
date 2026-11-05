"""Expand and physically order the 512 selected indexer groups."""
import ctypes
import os
from pathlib import Path
import torch

_lib = None
_tables = {}


def expand(chosen, positions, capacity):
    if chosen.ndim != 2 or chosen.shape[1] != 512 or not chosen.is_contiguous():
        return None
    if not positions.is_contiguous() or positions.numel() != chosen.shape[0]:
        return None
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(os.environ.get("FNX_INDEX_EXPAND_SO", str(Path(__file__).with_name("libfnx_index_expand.so"))))
        lib.fnx_expand_groups.argtypes = [ctypes.c_void_p]*5 + [ctypes.c_int]*2 + [ctypes.c_void_p]
        lib.fnx_expand_groups.restype = ctypes.c_int
        _lib = lib
    key = str(chosen.device)
    if key not in _tables:
        gather = torch.tensor([(i//4)*8 for i in range(2048)], dtype=torch.int32, device=chosen.device)
        offsets = torch.tensor([i%4 for i in range(2048)], dtype=torch.float32, device=chosen.device)
        _tables[key] = gather, offsets
    values = chosen.to(torch.float32)
    ordered = torch.empty((chosen.shape[0], 2056), dtype=torch.float32, device=chosen.device)
    rc = _lib.fnx_expand_groups(values.data_ptr(), positions.data_ptr(), ordered.data_ptr(),
                                *(t.data_ptr() for t in _tables[key]), chosen.shape[0], capacity,
                                torch.npu.current_stream(chosen.device).npu_stream)
    if rc:
        raise RuntimeError(f"index expansion launch rejected: {rc}")
    ordered = ordered[:, :2051].to(torch.int64)
    return ordered.clamp_max(capacity-1), ordered < capacity
