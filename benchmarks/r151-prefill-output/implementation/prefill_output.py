"""FP32 block sum -> BF16 token-major output; no recurrence or BMM changes."""
import ctypes,os
from pathlib import Path
import torch
_lib=None
def add_into(left,right,output,block,mode=4):
 if (left.shape!=(48,64,128) or right.shape!=left.shape or left.dtype!=torch.float32 or right.dtype!=torch.float32
     or output.ndim!=3 or output.shape[1:]!=(48,128) or output.dtype!=torch.bfloat16 or output.shape[0]%64
     or block<0 or (block+1)*64>output.shape[0] or left.device.type!='npu'
     or any(x.device!=left.device or not x.is_contiguous() for x in (left,right,output))):return False
 global _lib
 if _lib is None:
  _lib=ctypes.CDLL(os.environ.get('FNX_PREFILL_OUTPUT_SO',str(Path(__file__).with_name('libfnx_prefill_output.so'))))
  _lib.fnx_prefill_output_into.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_void_p];_lib.fnx_prefill_output_into.restype=ctypes.c_int
 rc=_lib.fnx_prefill_output_into(left.data_ptr(),right.data_ptr(),output.data_ptr()+block*64*48*128*2,mode,torch.npu.current_stream(left.device).npu_stream)
 if rc:raise RuntimeError(f'prefill output launch rejected: {rc}')
 return True
