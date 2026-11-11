"""Fuse diagonal extraction and the original rounded MoE output composition."""
import ctypes,os
from pathlib import Path
import torch
_lib=None
def merge(cartesian,weights,shared,gate):
    tensors=(cartesian,weights,shared,gate)
    if (cartesian.shape!=(10,25600) or weights.shape!=(1,10) or shared.shape!=(1,2560)
        or gate.numel()!=1 or cartesian.device.type!='npu'
        or any(t.dtype!=torch.bfloat16 or not t.is_contiguous() or t.device!=cartesian.device for t in tensors)):
        return None
    global _lib
    if _lib is None:
        lib=ctypes.CDLL(os.environ.get('FNX_MOE_EPILOGUE_SO',str(Path(__file__).with_name('libfnx_moe_epilogue.so'))))
        lib.fnx_moe_epilogue.argtypes=[ctypes.c_void_p]*6
        lib.fnx_moe_epilogue.restype=ctypes.c_int
        _lib=lib
    out=torch.empty_like(shared)
    rc=_lib.fnx_moe_epilogue(*(t.data_ptr() for t in (*tensors,out)),torch.npu.current_stream(cartesian.device).npu_stream)
    if rc:raise RuntimeError(f'MoE epilogue launch rejected: {rc}')
    return out
