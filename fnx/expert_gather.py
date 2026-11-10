"""Gather selected expert-major W4 weights/scales into the decoder's K-major layout."""
import ctypes
import os
from pathlib import Path
import torch

_lib = None

def gather(projection, ids):
    if (not projection.expert_major or projection._scales is None
            or (projection.K, projection.G, projection.N, projection.E) not in ((2560, 40, 1280, 512), (640, 10, 2560, 512))
            or ids.shape != (10,) or ids.dtype != torch.int64 or not ids.is_contiguous()):
        return None
    w, s = projection.w4, projection._scales
    if (w.dtype != torch.int32 or s.dtype != torch.bfloat16
            or not w.is_contiguous() or not s.is_contiguous()
            or w.device != s.device or w.device != ids.device or w.device.type != 'npu'):
        return None
    global _lib
    if _lib is None:
        lib = ctypes.CDLL(os.environ.get('FNX_EXPERT_GATHER_SO',str(Path(__file__).with_name('libfnx_expert_gather.so'))))
        lib.fnx_gather_pair.argtypes = [ctypes.c_void_p]*5+[ctypes.c_int]*3+[ctypes.c_void_p]
        lib.fnx_gather_pair.restype = ctypes.c_int
        _lib = lib
    ow = torch.empty((projection.K,10*projection.N//8),dtype=torch.int32,device=w.device)
    oscale = torch.empty((projection.G,10*projection.N),dtype=torch.bfloat16,device=w.device)
    rc = _lib.fnx_gather_pair(w.data_ptr(),s.data_ptr(),ids.data_ptr(),ow.data_ptr(),oscale.data_ptr(),
                              projection.K,projection.G,projection.N,torch.npu.current_stream(w.device).npu_stream)
    if rc:
        raise RuntimeError(f'expert gather launch rejected: {rc}')
    return ow, oscale
