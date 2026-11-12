"""Keep the original CANN 16-row reductions, combine only local epilogues."""
import ctypes,os
from pathlib import Path
import torch
from .norm_post import table
_lib=None
def library():
    global _lib
    if _lib is None:
        lib=ctypes.CDLL(os.environ.get('FNX_NORM_PAIR_SO',str(Path(__file__).with_name('libfnx_norm_pair.so'))))
        lib.fnx_norm_pair.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p];lib.fnx_norm_pair.restype=ctypes.c_int
        _lib=lib
    return _lib
def run(values,*,joint_reduce=False,scale_query=True):
    if values.shape!=(1,10240) or values.dtype!=torch.bfloat16 or values.device.type!='npu' or not values.is_contiguous():return None
    # Each BF16 square is unchanged. Slice dimension zero so each original
    # reduction retains shape [1,16,128] and strides [2048,128,1].
    squares=(values[:,:4096]*values[:,:4096]).view(2,16,128)
    if joint_reduce:
        sums=squares.view(1,32,128).sum(-1,keepdim=True);qsum,ksum=sums[:,:16],sums[:,16:]
    else:
        qsum=squares[0:1].sum(-1,keepdim=True);ksum=squares[1:2].sum(-1,keepdim=True)
    output=torch.empty((1,32,128),device=values.device,dtype=values.dtype)
    rc=library().fnx_norm_pair(values.data_ptr(),qsum.data_ptr(),ksum.data_ptr(),table(values.device).data_ptr(),output.data_ptr(),int(scale_query),torch.npu.current_stream(values.device).npu_stream)
    if rc:raise RuntimeError(f'q/k post pair launch rejected: {rc}')
    return output[:,:16],output[:,16:]
