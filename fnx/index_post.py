"""Exact FP32 index-score scaling and causal masking; optional four-head sum."""
import ctypes,os
from pathlib import Path
import torch
_lib=None
def run(values,positions,heads=1,mode=1):
    if (heads not in (1,4) or mode not in (0,1) or values.ndim!=2 or positions.ndim!=1
        or values.shape[0]!=heads*positions.numel() or positions.numel()<1
        or values.shape[1]<2048 or values.shape[1]%2048
        or values.dtype!=torch.float32 or positions.dtype!=torch.int64
        or values.device.type!='npu' or values.device!=positions.device
        or not values.is_contiguous() or not positions.is_contiguous()):return None
    global _lib
    if _lib is None:
        lib=ctypes.CDLL(os.environ.get('FNX_INDEX_POST_SO',str(Path(__file__).with_name('libfnx_index_post.so'))))
        lib.fnx_index_post.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*4+[ctypes.c_void_p]
        lib.fnx_index_post.restype=ctypes.c_int;_lib=lib
    tokens,groups=positions.numel(),values.shape[1]
    out=torch.empty((tokens,groups),dtype=torch.float32,device=values.device)
    rc=_lib.fnx_index_post(values.data_ptr(),positions.data_ptr(),out.data_ptr(),tokens,groups,heads,mode,
                           torch.npu.current_stream(values.device).npu_stream)
    if rc:raise RuntimeError(f'index postprocessing launch rejected: {rc}')
    return out
