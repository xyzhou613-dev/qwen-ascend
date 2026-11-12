"""Fixed-shape FP32 DeltaNet state update and rounded BF16 output epilogues."""
import ctypes,os
from pathlib import Path
import torch
_lib=None

def _load():
 global _lib
 if _lib is None:
  _lib=ctypes.CDLL(os.environ.get('FNX_PREFILL_POST_SO',str(Path(__file__).with_name('libfnx_prefill_post.so'))))
  _lib.fnx_prefill_state.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_void_p];_lib.fnx_prefill_state.restype=ctypes.c_int
  _lib.fnx_prefill_output.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_void_p];_lib.fnx_prefill_output.restype=ctypes.c_int
 return _lib

def state_update(current,decay,delta,tile=2048):
 if (current.shape!=(48,128,128) or delta.shape!=current.shape or decay.shape!=(48,1,1)
     or any(x.dtype!=torch.float32 or x.device!=current.device or not x.is_contiguous() for x in (current,decay,delta)) or current.device.type!='npu'):return None
 out=torch.empty_like(current)
 rc=_load().fnx_prefill_state(current.data_ptr(),decay.data_ptr(),delta.data_ptr(),out.data_ptr(),tile,torch.npu.current_stream(current.device).npu_stream)
 if rc:raise RuntimeError(f'FP32 state epilogue rejected: {rc}')
 return out

def output_add(left,right,tile=2048):
 if (left.shape!=(48,64,128) or right.shape!=left.shape or any(x.dtype!=torch.float32 or x.device!=left.device or not x.is_contiguous() for x in (left,right)) or left.device.type!='npu'):return None
 out=torch.empty_like(left,dtype=torch.bfloat16)
 rc=_load().fnx_prefill_output(left.data_ptr(),right.data_ptr(),out.data_ptr(),tile,torch.npu.current_stream(left.device).npu_stream)
 if rc:raise RuntimeError(f'BF16 output epilogue rejected: {rc}')
 return out
