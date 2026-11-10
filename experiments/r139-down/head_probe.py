import torch,torch_npu,json,statistics,gc
from pathlib import Path
from safetensors import safe_open
P=Path(__file__).resolve().parent
def bench(g):
 v=[]
 for _ in range(7):
  a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
  for _ in range(8):g.replay()
  b.record();b.synchronize();v.append(a.elapsed_time(b)/8/4)
 return statistics.median(v)
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.set_num_threads(16);torch.manual_seed(139139)
 with safe_open('/models/Qwen3.8-Flash-Next-Q/nonlayer.safetensors',framework='pt',device='cpu') as f:q=f.get_tensor('lm_head.weight.q8');s=f.get_tensor('lm_head.weight.s8')
 n,k=q.shape;group=k//s.shape[-1];raw=(q.to(torch.bfloat16).view(n,-1,group)*s.to(torch.bfloat16).unsqueeze(-1)).view(n,k).to('npu:0')
 x=torch.randn(1,k,device='npu:0',dtype=torch.bfloat16)*.03;records=[]
 for chunk in (62080,31040,15520,65504,0,62080):
  if chunk:parts=[(q[i:i+chunk].t().contiguous().to(x.device),s[i:i+chunk].t().contiguous().to(device=x.device,dtype=x.dtype),i,min(i+chunk,n)) for i in range(0,n,chunk)]
  def run():
   if chunk==0:return torch.nn.functional.linear(x,raw)
   y=torch.empty(1,n,device=x.device,dtype=x.dtype)
   for a,b,i,j in parts:y[:,i:j]=torch_npu.npu_weight_quant_batchmatmul(x,a,b,antiquant_group_size=group)
   return y
  run();torch.npu.synchronize();g=torch.npu.NPUGraph()
  with torch.npu.graph(g):
   for _ in range(4):out=run()
  exact=True;max_abs=0.;max_rel=0.;argmax_match=True
  for step in range(16):
   x.copy_(torch.randn_like(x)*(.03 if step%2 else .3));expected=torch.nn.functional.linear(x,raw);g.replay();torch.npu.synchronize()
   exact &= torch.equal(expected.view(torch.int16),out.view(torch.int16));max_abs=max(max_abs,(expected.float()-out.float()).abs().max().item());max_rel=max(max_rel,((expected.float()-out.float()).norm()/expected.float().norm()).item());argmax_match &= torch.equal(expected.argmax(-1),out.argmax(-1))
  row=dict(chunk_n=chunk,ms=bench(g),exact_vs_raw=exact,max_abs=max_abs,rel_l2=max_rel,argmax_match=argmax_match);records.append(row);(P/'head-probe.json').write_text(json.dumps(records,indent=2));print(json.dumps(row),flush=True)
  del g,out
  if chunk:del parts
  gc.collect();torch.npu.empty_cache()
 print('HEAD_COMPLETE',flush=True)
if __name__=='__main__':main()
