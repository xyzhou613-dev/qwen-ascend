"""Experimental per-output-channel A8W8 head from the existing grouped W8 head.

No calibration claim: this adds requantization and requires full model quality.
"""
import torch,torch_npu

class QuantHead:
    def __init__(self):self.chunks=[];self.N=0;self.K=0

    @classmethod
    @torch.no_grad()
    def from_grouped(cls,source,guard):
        assert source.chunks and len(source.chunks)==4
        obj=cls();obj.N=source.N;obj.K=source.K
        sums=torch.zeros(2,device='npu',dtype=torch.float32)
        for weight,scales,start,end in source.chunks:
            guard();n=end-start
            q=torch.empty((n,obj.K),device=weight.device,dtype=torch.int8)
            out_scale=torch.empty(n,device=weight.device,dtype=torch.float32)
            for first in range(0,n,256):
                if first%4096==0:guard()
                last=min(first+256,n)
                w=weight[:,first:last].t().contiguous().to(torch.bfloat16)
                scale=scales[:,first:last].t().contiguous().repeat_interleave(source.gs,1)
                reference=(w*scale).float()
                s=reference.abs().amax(1).clamp_min(1e-12)/127
                integer=(reference/s[:,None]).round().clamp(-127,127).to(torch.int8)
                q[first:last].copy_(integer);out_scale[first:last].copy_(s)
                reconstructed=integer.float()*s[:,None]
                sums[0].add_(reference.square().sum());sums[1].add_((reference-reconstructed).square().sum())
            obj.chunks.append((q.t(),out_scale,start,end))
        obj.weight_fit_relative_l2=float((sums[1]/sums[0].clamp_min(1e-12)).sqrt())
        obj.weight_storage_bytes=sum(w.numel()*w.element_size()+s.numel()*s.element_size() for w,s,_,_ in obj.chunks)
        return obj

    def __call__(self,x):
        assert x.ndim==2 and x.dtype==torch.bfloat16 and x.shape[1]==self.K
        q,scale=torch_npu.npu_dynamic_quant(x)
        out=torch.empty((x.shape[0],self.N),device=x.device,dtype=torch.bfloat16)
        for weight,weight_scale,start,end in self.chunks:
            out[:,start:end]=torch_npu.npu_quant_matmul(q,weight,weight_scale,
                pertoken_scale=scale,output_dtype=torch.bfloat16)
        return out
