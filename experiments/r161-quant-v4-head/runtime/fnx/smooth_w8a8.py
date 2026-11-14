"""Fixed-alpha SmoothQuant experiment; requires separate calibration inputs."""
import os
import torch,torch_npu
from safetensors import safe_open
from pathlib import Path
from .w8a8 import W8A8
P=Path(__file__).resolve().parent

class SmoothW8A8(W8A8):
    __slots__=('input_scale',)

    @classmethod
    @torch.no_grad()
    def from_bf16(cls,weight,input_amax):
        assert weight.ndim==2 and weight.dtype==torch.bfloat16
        obj=cls();obj.N,obj.K=weight.shape;obj.layout='nd'
        colmax=torch.zeros(obj.K,device=weight.device,dtype=torch.float32)
        for first in range(0,obj.N,256):
            colmax=torch.maximum(colmax,weight[first:first+256].float().abs().amax(0))
        amax=input_amax.to(device=weight.device,dtype=torch.float32)
        assert amax.shape==(obj.K,) and torch.isfinite(amax).all() and torch.all(amax>=0)
        # Fixed alpha=.5, fixed [.01,100] bound; no per-evaluation hyperparameter search.
        obj.input_scale=(colmax.clamp_min(1e-5)/amax.clamp_min(1e-5)).sqrt().clamp(.01,100).to(torch.bfloat16)
        obj.scale=torch.empty(obj.N,device=weight.device,dtype=torch.float32)
        q=torch.empty(obj.N,obj.K,device=weight.device,dtype=torch.int8)
        for first in range(0,obj.N,256):
            last=min(first+256,obj.N);wf=weight[first:last].float()/obj.input_scale.float()[None,:]
            scale=wf.abs().amax(1).clamp_min(1e-12)/127
            q[first:last].copy_((wf/scale[:,None]).round().clamp(-127,127).to(torch.int8));obj.scale[first:last].copy_(scale)
        obj.weight=q.t();return obj

    def __call__(self,x):
        if x.ndim<2 or x.shape[-1]!=self.K or x.dtype!=torch.bfloat16 or x.device!=self.weight.device:
            raise ValueError('SmoothW8A8 requires a same-device BF16 input with configured K')
        if x.numel()==0:return torch.empty((*x.shape[:-1],self.N),device=x.device,dtype=x.dtype)
        xq,xs=torch_npu.npu_dynamic_quant(x.reshape(-1,self.K),smooth_scales=self.input_scale)
        y=torch_npu.npu_quant_matmul(xq,self.weight,self.scale,pertoken_scale=xs,output_dtype=torch.bfloat16)
        return y.reshape(*x.shape[:-1],self.N)

def configure_layer(layer,raw_type):
    changed=[]
    directory=os.environ.get('FNX_W8A8_CALIB_DIR')
    if not directory:raise ValueError('FNX_DENSE_W8A8 requires frozen FNX_W8A8_CALIB_DIR statistics')
    with safe_open(str(Path(directory)/f'L{layer.idx:02d}.safetensors'),framework='pt',device='cpu') as f:
        if layer.ltype in ('linear','ple'):
            assert isinstance(layer.w_qkv,raw_type) and isinstance(layer.w_z,raw_type)
            weight=torch.cat((layer.w_qkv.w,layer.w_z.w),0)
            layer.w_qkvz=SmoothW8A8.from_bf16(weight,f.get_tensor('w_qkv'))
            layer.w_qkv=layer.w_z=None;layer.qkvz_ba=False;changed.append('w_qkvz')
            layer.w_out=SmoothW8A8.from_bf16(layer.w_out.w,f.get_tensor('w_out'));changed.append('w_out')
        else:
            for name in ('w_q','w_o'):
                weight=getattr(layer,name);assert isinstance(weight,raw_type)
                setattr(layer,name,SmoothW8A8.from_bf16(weight.w,f.get_tensor(name)));changed.append(name)
    return changed
