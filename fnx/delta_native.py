"""Exact ordered FP32 DeltaNet decode update on the supported 910B layout."""
import ctypes
import os
from pathlib import Path
import torch
import torch_npu

_lib = None

def _library():
    global _lib
    if _lib is None:
        path = os.environ.get('FNX_DN_NATIVE_SO', str(Path(__file__).with_name('libfnx_delta.so')))
        lib = ctypes.CDLL(path)
        lib.fnx_delta.argtypes = [ctypes.c_void_p]*7+[ctypes.c_int,ctypes.c_void_p]
        lib.fnx_delta.restype = ctypes.c_int
        if hasattr(lib, "fnx_delta_compact"):
            lib.fnx_delta_compact.argtypes = lib.fnx_delta.argtypes
            lib.fnx_delta_compact.restype = ctypes.c_int
        _lib = lib
    return _lib

def step(state, query, key, value, beta, decay, *, compact=False):
    if state.shape != (1,48,128,128) or state.dtype != torch.float32 or not state.is_contiguous():
        raise ValueError('native DeltaNet requires contiguous FP32 [1,48,128,128] state')
    if any(t.device != state.device for t in (query,key,value,beta,decay)):
        raise ValueError('native DeltaNet tensors must be on the same NPU')
    inputs = (query, key, value, beta)
    lib = _library()
    if compact:
        if not all(t.dtype == torch.bfloat16 and t.is_contiguous() for t in inputs):
            raise ValueError("compact DeltaNet requires contiguous BF16 inputs")
        expected = (2048, 2048, 6144, 48)
        kernel = lib.fnx_delta_compact
    else:
        inputs = tuple(t.float().contiguous() for t in inputs)
        expected = (6144, 6144, 6144, 48)
        kernel = lib.fnx_delta
    gate = decay.contiguous()
    if tuple(t.numel() for t in inputs) != expected or gate.numel() != 48 or gate.dtype != torch.float32:
        raise ValueError("unsupported native DeltaNet input shape or dtype")
    output = torch.empty((1,48,128),dtype=torch.float32,device=state.device)
    rc = kernel(state.data_ptr(), *(t.data_ptr() for t in inputs), gate.data_ptr(), output.data_ptr(),
                40, torch.npu.current_stream(state.device).npu_stream)
    if rc != 0:
        raise RuntimeError(f'native DeltaNet launch rejected: {rc}')
    return output
