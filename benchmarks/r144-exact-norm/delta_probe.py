"""Exact constant reuse/CSE probe; never loads Engine or any model weights."""
import json,gc
from pathlib import Path
import torch,torch_npu
from fnx import engine as E
from probe import compare,graph,timing
from delta_reuse import delta_chunk_batched_fp32 as candidate
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.manual_seed(1443);rows=[]
 for T in (1,63,64,65,128,512,2048):
  q=torch.randn(T,48,128,device='npu:0',dtype=torch.bfloat16)*.03;k=torch.randn_like(q)*.03;v=torch.randn_like(q)*.1
  g=-torch.rand(T,48,device='npu:0')*.2;beta=torch.rand(T,48,1,device='npu:0',dtype=torch.bfloat16)
  state=torch.randn(1,48,128,128,device='npu:0')*.01;states=[state.clone(),state.clone()];exact=[]
  for _ in range(3):
   ya,states[0]=E.delta_chunk_batched_fp32(q,k,v,g,beta,states[0]);yb,states[1]=candidate(q,k,v,g,beta,states[1]);torch.npu.synchronize()
   exact.append(dict(output=compare(ya,yb),state=compare(states[0],states[1])))
  ga,oa=graph(lambda:E.delta_chunk_batched_fp32(q,k,v,g,beta,state),1);gb,ob=graph(lambda:candidate(q,k,v,g,beta,state),1)
  times=[dict(mode=m,ms=timing(gr,4)) for m,gr in [('base',ga),('reuse',gb),('base',ga),('reuse',gb)]]
  row=dict(T=T,exact=exact,times=times);rows.append(row);P.joinpath('delta-probe.json').write_text(json.dumps(rows,indent=2));print('DELTA_REUSE',json.dumps(row),flush=True)
  assert all(r['output']['exact'] and r['state']['exact'] for r in exact)
  del ga,gb,oa,ob,q,k,v,g,beta,state,states;gc.collect();torch.npu.empty_cache()
 print('DELTA_REUSE_COMPLETE',flush=True)
if __name__=='__main__':main()
