"""Cached scales must match original scale arithmetic for changing experts."""
import os,time
import torch
from fnx import engine as E

@torch.no_grad()
def main():
    torch.npu.set_device(0)
    torch.manual_seed(56)
    for K,N in ((2560,1280),(640,2560)):
        G=K//64
        raw=torch.randint(-120,120,(G,512,N),dtype=torch.int8,device='npu:0')
        base=torch.rand(512,N,dtype=torch.bfloat16,device='npu:0')*.01
        # Only dimensions/storage metadata are needed to test scale selection.
        weight=torch.zeros(512,K,N//8,dtype=torch.int32,device='npu:0')
        os.environ['FNX_W4_SCALE_CACHE']='0';old=E.ExpW4(weight,raw,base,64,expert_major=True)
        os.environ['FNX_W4_SCALE_CACHE']='1';new=E.ExpW4(weight,raw,base,64,expert_major=True)
        full=raw.permute(1,0,2).to(torch.bfloat16)*base[:,None,:]
        assert torch.equal(full,new._scales)
        ids=torch.arange(10,device='npu:0',dtype=torch.int64)
        def timed(fn):
            fn();torch.npu.synchronize()
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph,capture_error_mode='relaxed'):out=fn()
            start=time.perf_counter()
            for _ in range(100):graph.replay()
            torch.npu.synchronize()
            return (time.perf_counter()-start)*10
        for _ in range(16):
            ids.copy_(torch.randperm(512,device='npu:0')[:10])
            assert torch.equal(old._scat(ids),new._scat(ids))
        print('SCALE',K,N,'old_ms',timed(lambda:old._scat(ids)),
              'new_ms',timed(lambda:new._scat(ids)),flush=True)
        del old,new,weight,raw,base,full
        torch.npu.empty_cache()
    print('SCALE_BITEXACT_ACCEPT',flush=True)

if __name__=='__main__':main()
