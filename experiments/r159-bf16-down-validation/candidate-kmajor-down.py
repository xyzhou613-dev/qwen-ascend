"""Decode-only direct GEMV over the production expert-major packed weights."""
import ctypes,os
from pathlib import Path
import torch
_lib=None
_rounded_lib=None
hits=0
def apply(projection,x,ids):
 if (not projection.expert_major or projection._scales is None or
     (projection.K,projection.N,projection.E,projection.gs)!=(640,2560,512,64) or
     tuple(x.shape)!=(10,640) or tuple(ids.shape)!=(10,) or
     x.dtype!=torch.bfloat16 or ids.dtype!=torch.int64 or
     projection.w4.dtype!=torch.int32 or projection._scales.dtype!=torch.bfloat16 or
     not all(t.is_contiguous() for t in (x,ids,projection.w4,projection._scales)) or
     x.device.type!='npu' or any(t.device!=x.device for t in (ids,projection.w4,projection._scales))):return None
 global _lib,_rounded_lib
 if _lib is None:
  _lib=ctypes.CDLL(os.environ.get('FNX_KMAJOR_SO',str(Path(__file__).with_name('libfnx_w4_kmajor.so'))))
  _lib.fnx_w4_kmajor.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p];_lib.fnx_w4_kmajor.restype=ctypes.c_int
  _lib.fnx_diag_epilogue.argtypes=[ctypes.c_void_p]*6;_lib.fnx_diag_epilogue.restype=ctypes.c_int
 if _rounded_lib is None:
  _rounded_lib=ctypes.CDLL(str(Path(__file__).with_name('libfnx_w4_bf16w.so')))
  _rounded_lib.fnx_w4_bf16w.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p]
  _rounded_lib.fnx_w4_bf16w.restype=ctypes.c_int
 out=torch.empty((10,2560),device=x.device,dtype=torch.bfloat16)
 rc=_rounded_lib.fnx_w4_bf16w(projection.w4.data_ptr(),projection._scales.data_ptr(),ids.data_ptr(),x.data_ptr(),out.data_ptr(),int(os.environ.get("FNX_BF16_DOWN_TILE","256")),torch.npu.current_stream(x.device).npu_stream)
 if rc:raise RuntimeError(f'kmajor down launch rejected: {rc}')
 return out
def merge(projection,x,ids,weights,shared,gate):
 if (tuple(weights.shape)!=(1,10) or tuple(shared.shape)!=(1,2560) or gate.numel()!=1 or
     any(t.dtype!=torch.bfloat16 or not t.is_contiguous() or t.device!=x.device for t in (weights,shared,gate))):return None
 diagonal=apply(projection,x,ids)
 if diagonal is None:return None
 out=torch.empty_like(shared)
 rc=_lib.fnx_diag_epilogue(diagonal.data_ptr(),weights.data_ptr(),shared.data_ptr(),gate.data_ptr(),out.data_ptr(),torch.npu.current_stream(x.device).npu_stream)
 if rc:raise RuntimeError(f'diagonal epilogue launch rejected: {rc}')
 global hits
 hits+=1
 return out
