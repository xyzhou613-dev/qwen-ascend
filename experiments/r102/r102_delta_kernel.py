"""Fuse pointwise state update only; both reductions remain CANN's original."""
import logging
_level=logging.getLogger().level
import pypto
logging.getLogger().setLevel(_level)
logging.getLogger('torch_npu.npugraph.__cudagraphs').setLevel(logging.WARNING)

@pypto.frontend.jit()
def update_product(sd:pypto.Tensor((48,128,128),pypto.DT_FP32),
                   key:pypto.Tensor((48,128),pypto.DT_BF16),
                   query:pypto.Tensor((48,128),pypto.DT_BF16),
                   value:pypto.Tensor((48,128),pypto.DT_BF16),
                   memory:pypto.Tensor((48,128),pypto.DT_FP32),
                   beta:pypto.Tensor((48,1),pypto.DT_BF16),
                   state:pypto.Tensor((48,128,128),pypto.DT_FP32),
                   product:pypto.Tensor((48,128,128),pypto.DT_FP32)):
    pypto.set_vec_tile_shapes(1,32,128)
    delta=pypto.mul(pypto.sub(pypto.cast(value,pypto.DT_FP32),memory),pypto.cast(beta,pypto.DT_FP32))
    outer=pypto.mul(pypto.unsqueeze(pypto.cast(key,pypto.DT_FP32),2),pypto.unsqueeze(delta,1))
    updated=pypto.add(sd,outer)
    result=pypto.mul(updated,pypto.unsqueeze(pypto.cast(query,pypto.DT_FP32),2))
    state[:]=updated
    product[:]=result
