"""Opt-in per-channel W8 / per-token A8 dense projections.

This changes the numerical path and needs model-quality acceptance. ND is the
default so ordinary startup does not initialize the legacy GE format caster.
NZ weights may be prepared in an isolated offline/diagnostic environment.
"""
import os
import torch
import torch_npu


class W8A8:
    __slots__ = ('weight', 'scale', 'K', 'N', 'layout')

    @classmethod
    @torch.no_grad()
    def from_bf16(cls, weight, *, layout='nd'):
        if weight.ndim != 2 or weight.dtype != torch.bfloat16 or weight.device.type != 'npu':
            raise ValueError('W8A8 construction requires a BF16 NPU [N,K] weight')
        if layout not in ('nd', 'nz'):
            raise ValueError('W8A8 layout must be nd or nz')
        obj=cls();obj.N,obj.K=weight.shape;obj.layout=layout
        obj.scale=torch.empty(obj.N,device=weight.device,dtype=torch.float32)
        q=torch.empty((obj.N,obj.K),device=weight.device,dtype=torch.int8)
        # Bound startup scratch even while an original BF16 projection lives.
        for first in range(0,obj.N,256):
            last=min(first+256,obj.N);wf=weight[first:last].float()
            scale=wf.abs().amax(1).clamp_min(1e-12)/127
            obj.scale[first:last].copy_(scale)
            q[first:last].copy_((wf/scale[:,None]).round().clamp(-127,127).to(torch.int8))
            del wf,scale
        # A transposed ND view was faster than a contiguous ND K,N tensor.
        obj.weight=q.t()
        if layout=='nz':
            previous=torch.npu.config.allow_internal_format
            try:
                torch.npu.config.allow_internal_format=True
                obj.weight=torch_npu.npu_format_cast(obj.weight.contiguous(),29)
            finally:
                torch.npu.config.allow_internal_format=previous
            if torch_npu.get_npu_format(obj.weight)!=29:
                raise RuntimeError('NZ weight conversion did not produce format 29')
        return obj

    def __call__(self,x):
        if x.ndim<2 or x.shape[-1]!=self.K or x.dtype!=torch.bfloat16 or x.device!=self.weight.device:
            raise ValueError('W8A8 requires same-device BF16 input with the configured K')
        shape=x.shape
        x2=x.reshape(-1,self.K)
        if x2.shape[0]==0:
            return torch.empty((*shape[:-1],self.N),device=x.device,dtype=x.dtype)
        xq,xscale=torch_npu.npu_dynamic_quant(x2)
        y=torch_npu.npu_quant_matmul(xq,self.weight,self.scale,
            pertoken_scale=xscale,output_dtype=torch.bfloat16)
        return y.reshape(*shape[:-1],self.N)


def configure_layer(layer, raw_type):
    """Replace only measured large Raw attention projections, releasing BF16.

    Return a manifest of actual replacements; an empty list is an error to the
    experiment launcher when a selected layer was expected to be converted.
    Small mixer, router, q/k/v-cache, normalization and FP32 states are retained.
    """
    layout=os.environ.get('FNX_W8A8_LAYOUT','nd')
    quant_out=os.environ.get('FNX_W8A8_QUANT_OUT','1')=='1'
    changed=[]
    if layer.ltype in ('linear','ple'):
        if isinstance(layer.w_qkv,raw_type) and isinstance(layer.w_z,raw_type):
            weight=torch.cat((layer.w_qkv.w,layer.w_z.w),0)
            layer.w_qkvz=W8A8.from_bf16(weight,layout=layout)
            layer.w_qkv=layer.w_z=None
            layer.qkvz_ba=False
            changed.append('w_qkvz')
        if quant_out and isinstance(layer.w_out,raw_type):
            layer.w_out=W8A8.from_bf16(layer.w_out.w,layout=layout)
            changed.append('w_out')
    elif layer.ltype=='full':
        # Keep the small k/v projections in their existing BF16 path.
        for name in (('w_q','w_o') if quant_out else ('w_q',)):
            weight=getattr(layer,name)
            if isinstance(weight,raw_type):
                setattr(layer,name,W8A8.from_bf16(weight.w,layout=layout))
                changed.append(name)
    return changed
