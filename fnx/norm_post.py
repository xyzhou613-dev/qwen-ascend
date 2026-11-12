"""BF16 l2norm postprocessing; retain original Torch square/ReduceSum."""
import ctypes,os
from pathlib import Path
import torch
_lib=None
_tables={}
def table(device):
 key=str(device)
 if key not in _tables:
  bits=torch.arange(65536,dtype=torch.int32,device=device).to(torch.int16)
  _tables[key]=torch.rsqrt(bits.view(torch.bfloat16)+1e-6)
 return _tables[key]
def run(x):
 global _lib
 if x.device.type!='npu' or x.dtype!=torch.bfloat16 or x.ndim!=3 or x.shape[1:]!=(16,128) or x.stride(0)<2048 or x.stride(1)!=128 or x.stride(2)!=1:return None
 if _lib is None:
  _lib=ctypes.CDLL(os.environ.get('FNX_NORM_POST_SO',str(Path(__file__).with_name('libfnx_norm_post.so'))))
  _lib.fnx_norm_lookup.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p];_lib.fnx_norm_lookup.restype=ctypes.c_int
 sums=(x*x).sum(-1,keepdim=True)
 out=torch.empty(x.shape,dtype=x.dtype,device=x.device)
 rc=_lib.fnx_norm_lookup(x.data_ptr(),sums.data_ptr(),table(x.device).data_ptr(),out.data_ptr(),x.shape[0]*16,x.stride(0),torch.npu.current_stream(x.device).npu_stream)
 if rc:raise RuntimeError(f'norm post launch rejected: {rc}')
 return out
