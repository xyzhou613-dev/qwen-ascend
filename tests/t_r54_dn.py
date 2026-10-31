"""Batched delta preparation versus the stable sequential 64-token solver."""
import os,time,json
import torch
from fnx import engine as E

def timed(fn):
    fn();torch.npu.synchronize()
    stream=torch.npu.Stream()
    with torch.npu.stream(stream):
        g=torch.npu.NPUGraph()
        with torch.npu.graph(g,stream=stream,capture_error_mode='relaxed'):fn()
    torch.npu.synchronize();start=time.perf_counter()
    for _ in range(5):g.replay()
    torch.npu.synchronize()
    return (time.perf_counter()-start)*200

@torch.no_grad()
def main():
    dev='npu:1';torch.npu.set_device(dev);torch.manual_seed(54)
    for T in (65,512,2048,4096):
        for correlated in (False,True):
            q=torch.randn(T,E.NVH,E.HKD,device=dev)
            k=(torch.ones_like(q)*.5+q*.01) if correlated else torch.randn_like(q)
            q=torch.nn.functional.normalize(q,dim=-1).to(torch.bfloat16)/(E.HKD**.5)
            k=torch.nn.functional.normalize(k,dim=-1).to(torch.bfloat16)
            v=torch.randn(T,E.NVH,E.HVD,device=dev,dtype=torch.bfloat16)
            lg=-torch.rand(T,E.NVH,device=dev)*.15
            beta=torch.ones(T,E.NVH,device=dev) if correlated else torch.rand(T,E.NVH,device=dev)
            state=torch.randn(1,E.NVH,E.HKD,E.HVD,device=dev)*.01
            os.environ['FNX_DELTA_BATCH']='0';ref,rs=E.delta_chunk_fp32(q,k,v,lg,beta,state)
            os.environ['FNX_DELTA_BATCH']='1';got,gs=E.delta_chunk_fp32(q,k,v,lg,beta,state)
            rel=float((ref.float()-got.float()).abs().mean()/(ref.float().abs().mean()+1e-9))
            sr=float((rs-gs).abs().mean()/(rs.abs().mean()+1e-9))
            print('DELTA_COMPARE',T,correlated,rel,sr,flush=True)
            assert torch.isfinite(got).all() and torch.isfinite(gs).all() and rel<1e-4 and sr<1e-4
            if not correlated:
                os.environ['FNX_DELTA_BATCH']='0';old=timed(lambda:E.delta_chunk_fp32(q,k,v,lg,beta,state))
                os.environ['FNX_DELTA_BATCH']='1';new=timed(lambda:E.delta_chunk_fp32(q,k,v,lg,beta,state))
                print('DELTA_SPEED',T,old,new,flush=True)
    print('DELTA_BATCH_ACCEPT',flush=True)

if __name__=='__main__':main()
