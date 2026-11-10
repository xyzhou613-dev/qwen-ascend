import torch,torch_npu,json,statistics
from pathlib import Path
from safetensors import safe_open
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.set_num_threads(16);torch.manual_seed(139141);d='npu:0'
 with safe_open('/models/Qwen3.8-Flash-Next-Q/nonlayer.safetensors',framework='pt',device='cpu') as f:q=f.get_tensor('lm_head.weight.q8')[:62080];s=f.get_tensor('lm_head.weight.s8')[:62080]
 n,k=q.shape;group=k//s.shape[-1];qt=q.t().contiguous().to(d);ss=s.t().contiguous().to(device=d,dtype=torch.bfloat16)
 raw_cpu=(q.to(torch.bfloat16).view(n,-1,group)*s.to(torch.bfloat16).unsqueeze(-1)).view(n,k).to(d)
 eye=torch.eye(k,device=d,dtype=torch.bfloat16)
 raw_cann=torch_npu.npu_weight_quant_batchmatmul(eye,qt,ss,antiquant_group_size=group).t().contiguous();torch.npu.synchronize()
 stats=dict(weight_equal=torch.equal(raw_cpu.view(torch.int16),raw_cann.view(torch.int16)),weight_diff_count=int((raw_cpu!=raw_cann).sum().item()),weight_max_diff=(raw_cpu.float()-raw_cann.float()).abs().max().item(),outputs={})
 for step in range(16):
  x=torch.randn(1,k,device=d,dtype=torch.bfloat16)*(.03 if step%2 else .3)
  expected=torch_npu.npu_weight_quant_batchmatmul(x,qt,ss,antiquant_group_size=group)
  for name,w in (('cpu_deq',raw_cpu),('cann_deq',raw_cann)):
   out=torch.nn.functional.linear(x,w);torch.npu.synchronize();r=stats['outputs'].setdefault(name,dict(exact=True,max_abs=0.,different=0))
   r['exact'] &= torch.equal(expected.view(torch.int16),out.view(torch.int16));r['max_abs']=max(r['max_abs'],(expected.float()-out.float()).abs().max().item());r['different']+=int((expected!=out).sum().item())
 (P/'head-deq-probe.json').write_text(json.dumps(stats,indent=2));print(json.dumps(stats),flush=True)
 print('HEAD_DEQ_COMPLETE',flush=True)
if __name__=='__main__':main()
