"""Identify the exact FP32 middle-axis summation used by CANN."""
from pathlib import Path
import json
import torch,torch_npu

def tree(x,adjacent):
    while x.shape[-2]>1:
        n=x.shape[-2]
        x=x[...,0:n:2,:]+x[...,1:n:2,:] if adjacent else x[...,:n//2,:]+x[...,n//2:,:]
    return x.squeeze(-2)

def lanes(x,n,finish):
    acc=x[...,:n,:].clone()
    for begin in range(n,x.shape[-2],n):acc=acc+x[...,begin:begin+n,:]
    if finish=='adjacent':return tree(acc,True)
    if finish=='halves':return tree(acc,False)
    out=acc[...,0,:].clone()
    for i in range(1,n):out=out+acc[...,i,:]
    return out

@torch.no_grad()
def main():
    torch.npu.set_device(0);records=[]
    for seed in range(4):
        torch.manual_seed(107107+seed)
        x=torch.randn(1,48,128,128,device='npu:0')*.01
        k=torch.randn(1,48,128,1,dtype=torch.bfloat16,device=x.device)*.1
        x=x*k
        expected=x.sum(-2).cpu();data=x.cpu()
        methods={'tree_adjacent':lambda:tree(data,True),'tree_halves':lambda:tree(data,False)}
        for n in (1,2,4,8,16,32,64,128):
            for finish in ('sequential','adjacent','halves'):
                methods[f'lanes{n}_{finish}']=lambda n=n,finish=finish:lanes(data,n,finish)
        result={}
        for name,fn in methods.items():
            out=fn();result[name]={'exact':torch.equal(out,expected),'equal_fraction':(out==expected).float().mean().item(),'max_abs':(out-expected).abs().max().item()}
        print('REDUCE_ORDER',seed,json.dumps({k:v for k,v in result.items() if v['exact'] or v['equal_fraction']>.8}),flush=True)
        records.append(result)
    p=Path('/root/ascend-dev/r107');p.mkdir(exist_ok=True);(p/'reduce-order.json').write_text(json.dumps(records,indent=2))
    print('R107_REDUCE_COMPLETE',flush=True)

if __name__=='__main__':main()
