"""Compiler probes; retain every original BF16 rounding boundary."""
import logging
_level=logging.getLogger().level
import pypto
logging.getLogger().setLevel(_level)
logging.getLogger('torch_npu.npugraph.__cudagraphs').setLevel(logging.WARNING)

@pypto.frontend.jit()
def combine8192(tokens:pypto.Tensor((81920,2560),pypto.DT_BF16),
                inverse:pypto.Tensor((8192,10),pypto.DT_INT32),
                weights:pypto.Tensor((8192,10,1),pypto.DT_BF16),
                output:pypto.Tensor((8192,2560),pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(1,10,256)
    restored=pypto.index_select(tokens,0,inverse)
    product=pypto.cast(pypto.mul(pypto.cast(restored,pypto.DT_FP32),pypto.cast(weights,pypto.DT_FP32)),pypto.DT_BF16,pypto.CastMode.CAST_RINT)
    total=pypto.sum(pypto.cast(product,pypto.DT_FP32),1)
    output[:]=pypto.cast(total,pypto.DT_BF16,pypto.CastMode.CAST_RINT)

@pypto.frontend.jit()
def conv_step(history:pypto.Tensor((10240,3),pypto.DT_BF16),
              current:pypto.Tensor((10240,1),pypto.DT_BF16),
              weights:pypto.Tensor((10240,4),pypto.DT_BF16),
              new_history:pypto.Tensor((10240,3),pypto.DT_BF16),
              output:pypto.Tensor((10240,),pypto.DT_BF16)):
    pypto.set_vec_tile_shapes(256,4)
    buffer=pypto.concat([history,current],-1)
    product=pypto.cast(pypto.mul(pypto.cast(buffer,pypto.DT_FP32),pypto.cast(weights,pypto.DT_FP32)),pypto.DT_BF16,pypto.CastMode.CAST_RINT)
    total=pypto.sum(pypto.cast(product,pypto.DT_FP32),1)
    next_history=pypto.concat([history[:,1:3],current],-1)
    output[:]=pypto.cast(total,pypto.DT_BF16,pypto.CastMode.CAST_RINT)
    new_history[:]=next_history
