"""Sparse in-place writes: no full KV-sized ScatterElements workspace."""
import torch,torch_npu
def kv(var,updates,pos):
    b,heads=updates.shape[:2]
    rows=torch.arange(b,device=pos.device,dtype=torch.int64).repeat_interleave(heads)
    h=torch.arange(heads,device=pos.device,dtype=torch.int64).repeat(b)
    indices=torch.stack((rows,h,pos.repeat_interleave(heads)),1)
    torch_npu.npu_scatter_nd_update_(var,indices,updates.reshape(b*heads,-1).contiguous())
def index(var,updates,pos):
    rows=torch.arange(updates.shape[0],device=pos.device,dtype=torch.int64)
    torch_npu.npu_scatter_nd_update_(var,torch.stack((rows,pos),1),updates.contiguous())
