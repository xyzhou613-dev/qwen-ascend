"""Choose bounded FP32 preparation size using captured device execution."""
import os,time,gc,json
from pathlib import Path
import torch
from fnx import engine as E

@torch.no_grad()
def main():
    torch.npu.set_device(0);T=8192
    q=torch.randn(T,48,128,dtype=torch.bfloat16,device='npu:0')
    k=E.l2norm(torch.randn_like(q));q=E.l2norm(q)/128**.5
    v=torch.randn_like(q);g=-torch.rand(T,48,device=q.device)*.1
    beta=torch.rand(T,48,1,dtype=q.dtype,device=q.device)
    st=torch.randn(1,48,128,128,device=q.device)*.01
    os.environ['FNX_DELTA_WINDOW']='0';a,sa=E.delta_chunk_fp32(q,k,v,g,beta,st)
    rows=[]
    for window in (8192,4096,2048,1024,512):
        os.environ['FNX_DELTA_WINDOW']=str(window)
        b,sb=E.delta_chunk_fp32(q,k,v,g,beta,st);torch.npu.synchronize()
        exact=torch.equal(a,b) and torch.equal(sa,sb)
        if not exact:
            print('WINDOW_REJECT',window,flush=True);continue
        stream=torch.npu.Stream()
        with torch.npu.stream(stream):
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):
                out,state=E.delta_chunk_fp32(q,k,v,g,beta,st)
        torch.npu.synchronize();t=time.perf_counter()
        for _ in range(5):graph.replay()
        torch.npu.synchronize()
        row={'window':window,'ms':(time.perf_counter()-t)*200,'bitexact':exact}
        rows.append(row);print('WINDOW_PERF',row,flush=True)
        del graph,out,state,b,sb;gc.collect();torch.npu.empty_cache()
    # 8192 would not bound the peak as effectively; select among <=4096.
    best=min((row for row in rows if row['window']<=4096),key=lambda row:row['ms'])
    Path('/root/ascend-dev/r67/window.json').write_text(json.dumps({'best':best,'all':rows},indent=2))
    print('WINDOW_BENCH_ACCEPT',best,flush=True)

if __name__=='__main__':main()
