import os,time
import torch
from fnx import engine as E

@torch.no_grad()
def main():
    torch.npu.set_device(0);inv=E.inv_freq('npu:0')
    positions=torch.cat((torch.tensor([0,1,3,2047,2048,16383,65535,131071,262143],device=inv.device),
                         torch.randint(0,262144,(1024,),device=inv.device)))
    for dtype in (torch.bfloat16,torch.float32):
        for pos in (positions,positions[:1],positions[0]):
            os.environ['FNX_ROPE_CACHE']='0';a,b=E.rope_cs(pos,inv,dtype)
            os.environ['FNX_ROPE_CACHE']='1';c,d=E.rope_cs(pos,inv,dtype)
            assert torch.equal(a,c) and torch.equal(b,d),(dtype,pos.shape)
    print('ROPE_CACHE_BITEXACT_ACCEPT',flush=True)
    pos=positions[:1];stream=torch.npu.Stream();graphs=[]
    with torch.npu.stream(stream):
        for flag in ('0','1'):
            os.environ['FNX_ROPE_CACHE']=flag
            E.rope_cs(pos,inv);torch.npu.synchronize()
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):
                for _ in range(36):last=E.rope_cs(pos,inv)
            graphs.append((flag,graph,last))
    for flag,graph,last in graphs:
        start=time.perf_counter()
        for _ in range(100):graph.replay()
        torch.npu.synchronize()
        print('ROPE36_MS',flag,(time.perf_counter()-start)*10,flush=True)
        pos.fill_(260001);graph.replay();torch.npu.synchronize()
        os.environ['FNX_ROPE_CACHE']='0';ref=E.rope_cs(pos,inv)
        assert torch.equal(last[0],ref[0]) and torch.equal(last[1],ref[1])
    print('ROPE_GRAPH_ACCEPT',flush=True)

if __name__=='__main__':main()
