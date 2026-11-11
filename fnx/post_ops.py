"""Decode postprocessing with the original intermediate BF16 rounding."""
import ctypes
import os
from pathlib import Path
import torch
_lib=None
def _library():
    global _lib
    if _lib is None:
        lib=ctypes.CDLL(os.environ.get('FNX_POST_SO',str(Path(__file__).with_name('libfnx_post.so'))))
        lib.fnx_hc_activation.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]
        lib.fnx_hc_sigmoid_reduce.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int,ctypes.c_void_p]
        lib.fnx_post_mul.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]
        lib.fnx_moe_merge.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int,ctypes.c_void_p]
        for name in ('fnx_hc_activation','fnx_hc_sigmoid_reduce','fnx_post_mul','fnx_moe_merge'):getattr(lib,name).restype=ctypes.c_int
        _lib=lib
    return _lib
def _same(dtype,*tensors):
    return tensors[0].device.type=='npu' and all(t.dtype==dtype and t.is_contiguous() and t.device==tensors[0].device for t in tensors)
def _check(rc):
    if rc:raise RuntimeError(f'decode postprocessing launch rejected: {rc}')
def activate(di):
    if di.shape!=(1,1,324) or not _same(torch.bfloat16,di):return None
    mixed=torch.empty((1,1,320),device=di.device,dtype=di.dtype);gate=torch.empty((1,1,4),device=di.device,dtype=di.dtype)
    _check(_library().fnx_hc_activation(di.data_ptr(),mixed.data_ptr(),gate.data_ptr(),1,0,torch.npu.current_stream(di.device).npu_stream))
    return mixed,gate
def mix_reduce(values,logits):
    if values.shape!=(1,1,10240) or logits.shape!=values.shape or not _same(torch.bfloat16,values,logits):return None
    out=torch.empty((1,1,2560),device=values.device,dtype=values.dtype)
    _check(_library().fnx_hc_sigmoid_reduce(values.data_ptr(),logits.data_ptr(),out.data_ptr(),1,torch.npu.current_stream(values.device).npu_stream));return out
def mul_cast(values,weights,round_input=False):
    n=values.numel()
    if (values.shape[0]!=1 or n not in (6144,10240) or weights.numel()!=n or not _same(torch.float32,values,weights)):return None
    out=torch.empty_like(values,dtype=torch.bfloat16)
    _check(_library().fnx_post_mul(values.data_ptr(),weights.data_ptr(),out.data_ptr(),n,int(round_input),torch.npu.current_stream(values.device).npu_stream));return out
def moe_merge(experts,shared,gate):
    if experts.numel()!=2560 or shared.numel()!=2560 or gate.numel()!=1 or not _same(torch.bfloat16,experts,shared,gate):return None
    out=torch.empty_like(experts)
    _check(_library().fnx_moe_merge(experts.data_ptr(),shared.data_ptr(),gate.data_ptr(),out.data_ptr(),1,torch.npu.current_stream(experts.device).npu_stream));return out
