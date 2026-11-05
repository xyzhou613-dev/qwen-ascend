"""BF16 product rounding followed by exact grouped accumulation."""
import ctypes
import os
from pathlib import Path
import torch

_lib=None


def reduce(values, weights, *, inverse=None, mean=False, order=0):
    if mean:
        if values.shape != weights.shape or values.ndim!=3 or values.shape[1]!=4:
            return None
        batch,groups,width=values.shape
        if not values.is_contiguous() or not weights.is_contiguous():
            return None
        x=values.view(-1,width);w=weights;mode=0
    else:
        if weights.ndim!=2 or weights.shape[1]!=10:
            return None
        batch,groups=weights.shape;width=values.shape[-1]
        x=values.view(-1,width);w=weights.float().contiguous();mode=2 if inverse is not None else 1
    if width!=2560 or values.dtype!=torch.bfloat16 or weights.dtype!=torch.bfloat16 or x.stride(1)!=1:
        return None
    if x.device!=w.device:
        return None
    if inverse is not None and (inverse.dtype!=torch.int64 or not inverse.is_contiguous()
                                or inverse.numel()!=batch*groups or inverse.device!=x.device):
        return None
    global _lib
    if _lib is None:
        lib=ctypes.CDLL(os.environ.get("FNX_WEIGHTED_REDUCE_SO",str(Path(__file__).with_name("libfnx_reduce.so"))))
        lib.fnx_weighted_reduce.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*6+[ctypes.c_void_p]
        lib.fnx_weighted_reduce.restype=ctypes.c_int
        _lib=lib
    output=torch.empty((batch,width),dtype=torch.bfloat16,device=x.device)
    rc=_lib.fnx_weighted_reduce(x.data_ptr(),w.data_ptr(),inverse.data_ptr() if inverse is not None else 0,
                                output.data_ptr(),batch,groups,width,x.stride(0),mode,order,
                                torch.npu.current_stream(x.device).npu_stream)
    if rc:
        raise RuntimeError(f"weighted reduction launch rejected: {rc}")
    return output
