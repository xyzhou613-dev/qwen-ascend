"""Fixed FP32 [48,128,128] multiply-then-add, with independent rounding."""
import ctypes,os
from pathlib import Path
import torch
_lib=None
def run(current,decay,delta,mode=4):
 if (current.shape!=(48,128,128) or delta.shape!=current.shape or decay.shape!=(48,1,1)
     or current.device.type!='npu' or any(x.dtype!=torch.float32 or x.device!=current.device or not x.is_contiguous() for x in (current,decay,delta))):return None
 global _lib
 if _lib is None:
  _lib=ctypes.CDLL(os.environ.get('FNX_STATE_UPDATE_SO',str(Path(__file__).with_name('libfnx_state_update.so'))))
  _lib.fnx_state_update.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_void_p];_lib.fnx_state_update.restype=ctypes.c_int
 out=torch.empty_like(current)
 rc=_lib.fnx_state_update(current.data_ptr(),decay.data_ptr(),delta.data_ptr(),out.data_ptr(),mode,torch.npu.current_stream(current.device).npu_stream)
 if rc:raise RuntimeError(f'FP32 state update rejected: {rc}')
 return out
