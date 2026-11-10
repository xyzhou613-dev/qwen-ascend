"""Four-tap BF16 prefill convolution with exact product rounding."""
import ctypes,os
from pathlib import Path
import torch
_lib=None

def run(sequence,weights):
    if sequence.ndim!=2 or weights.shape!=(sequence.shape[0],4):
        raise ValueError('prefill convolution requires [D,T+3] and [D,4]')
    dimension,length=sequence.shape;tokens=length-3
    if dimension%4 or tokens<1 or sequence.device.type!='npu':
        raise ValueError('unsupported prefill convolution dimensions/device')
    if any(t.dtype!=torch.bfloat16 or not t.is_contiguous() or t.device!=sequence.device for t in (sequence,weights)):
        raise ValueError('prefill convolution requires contiguous BF16 tensors on one NPU')
    global _lib
    if _lib is None:
        lib=ctypes.CDLL(os.environ.get('FNX_CONV_PREFILL_SO',str(Path(__file__).with_name('libfnx_conv_prefill.so'))))
        lib.fnx_conv_prefill.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int]*2+[ctypes.c_void_p]
        lib.fnx_conv_prefill.restype=ctypes.c_int;_lib=lib
    out=torch.empty((dimension,tokens),device=sequence.device,dtype=torch.bfloat16)
    rc=_lib.fnx_conv_prefill(sequence.data_ptr(),weights.data_ptr(),out.data_ptr(),dimension,tokens,
                              torch.npu.current_stream(sequence.device).npu_stream)
    if rc:raise RuntimeError(f'prefill convolution launch rejected: {rc}')
    return out
