import torch,torch_npu,json,statistics,gc
from pathlib import Path
from safetensors import safe_open
P=Path(__file__).resolve().parent
def bench(g):
 a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);samples=[]
 for _ in range(5):
  a.record()
  for _ in range(8):g.replay()
  b.record();b.synchronize();samples.append(a.elapsed_time(b)/8/4)
 return statistics.median(samples)
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.set_num_threads(16);torch.manual_seed(139140)
 with safe_open('/models/Qwen3.8-Flash-Next-Q/nonlayer.safetensors',framework='pt',device='cpu') as f:q=f.get_tensor('lm_head.weight.q8');s=f.get_tensor('lm_head.weight.s8')
 n,k=q.shape;group=k//s.shape[-1];raw=(q.to(torch.bfloat16).view(n,-1,group)*s.to(torch.bfloat16).unsqueeze(-1)).view(n,k).to('npu:0')
 parts=[(q[i:i+62080].t().contiguous().to(raw.device),s[i:i+62080].t().contiguous().to(device=raw.device,dtype=raw.dtype),i,min(i+62080,n)) for i in range(0,n,62080)]
 x=torch.randn(1,k,device=raw.device,dtype=raw.dtype)*.03
 def base():
  y=torch.empty(1,n,device=x.device,dtype=x.dtype)
  for a,b,i,j in parts:y[:,i:j]=torch_npu.npu_weight_quant_batchmatmul(x,a,b,antiquant_group_size=group)
  return y
 records=[]
 for chunk in (62080,31040,32768,65504,8192,0):
  def run():
   if chunk==0:return torch.nn.functional.linear(x,raw)
   y=torch.empty(1,n,device=x.device,dtype=x.dtype)
   for i in range(0,n,chunk):y[:,i:i+chunk]=torch.nn.functional.linear(x,raw[i:i+chunk])
   return y
  run();torch.npu.synchronize();g=torch.npu.NPUGraph()
  with torch.npu.graph(g):
   for _ in range(4):out=run()
  exact=True;max_abs=0.;different=0;argmax=True
  for step in range(16):
   x.copy_(torch.randn_like(x)*(.03 if step%2 else .3));expected=base();g.replay();torch.npu.synchronize();d=(expected.float()-out.float()).abs();exact &= torch.equal(out.view(torch.int16),expected.view(torch.int16));max_abs=max(max_abs,d.max().item());different+=int((expected!=out).sum().item());argmax &= torch.equal(expected.argmax(-1),out.argmax(-1))
  row=dict(chunk=chunk,ms=bench(g),exact_vs_production=exact,max_abs=max_abs,different_elements=different,argmax_match=argmax);records.append(row);(P/'head-exact-probe.json').write_text(json.dumps(records,indent=2));print(json.dumps(row),flush=True)
  del g,out;gc.collect();torch.npu.empty_cache()
 print('HEAD_EXACT_COMPLETE',flush=True)
if __name__=='__main__':main()
