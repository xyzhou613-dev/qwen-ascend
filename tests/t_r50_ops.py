"""Small capability probes, run only after the production baseline completes."""
import time
import torch
import torch_npu


def bench(fn, n=10):
    fn(); torch.npu.synchronize()
    t=time.perf_counter()
    for _ in range(n): fn()
    torch.npu.synchronize()
    return (time.perf_counter()-t)*1e3/n


def main():
    dev='npu:1'
    torch.npu.set_device(dev)
    for dtype in (torch.float32,torch.bfloat16):
        try:
            T=64
            q=torch.randn(T,16,128,device=dev,dtype=torch.bfloat16)*.05
            k=torch.randn_like(q)*.05
            v=torch.randn(T,48,128,device=dev,dtype=torch.bfloat16)
            b=torch.full((T,48),.5,device=dev,dtype=torch.bfloat16)
            g=torch.full((T,48),-.1,device=dev,dtype=torch.float32)
            state=torch.zeros(1,48,128,128,device=dev,dtype=dtype)
            lens=torch.tensor([T],device=dev,dtype=torch.int32)
            fn=lambda: torch_npu.npu_chunk_gated_delta_rule(q,k,v,beta=b,g=g,initial_state=state,actual_seq_lengths=lens,scale=1.)
            o,s=fn(); torch.npu.synchronize()
            print('FUSED_DN',str(dtype),tuple(o.shape),s.dtype,bench(fn),flush=True)
        except Exception as ex: print('FUSED_DN_ERROR',str(dtype),str(ex)[:900],flush=True)
    for batch in (1,8,32):
        try:
            # Same per-group int4 format as production, now with a token batch.
            w=torch.randint(-8,8,(batch,256,128),device=dev,dtype=torch.int32)
            packed=torch_npu.npu_convert_weight_to_int4pack(w)
            x=torch.randn(batch,1,256,device=dev,dtype=torch.bfloat16)
            scale=torch.ones(batch,4,128,device=dev,dtype=torch.bfloat16)*.01
            fn=lambda: torch_npu.npu_weight_quant_batchmatmul(x,packed,scale,antiquant_group_size=64)
            out=fn(); torch.npu.synchronize()
            print('BATCH_W4',batch,tuple(out.shape),bench(fn),flush=True)
        except Exception as ex: print('BATCH_W4_ERROR',batch,str(ex)[:900],flush=True)
    try:
        a=torch.eye(32,device=dev).expand(48,32,32).contiguous()
        b=torch.randn(48,32,128,device=dev)
        fn=lambda: torch.linalg.solve_triangular(a,b,upper=False,unitriangular=True)
        print('SOLVE_TRI',bench(fn),flush=True)
    except Exception as ex: print('SOLVE_ERROR',str(ex)[:600],flush=True)


if __name__=='__main__':
    with torch.no_grad(): main()
