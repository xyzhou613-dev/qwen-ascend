"""Projection and true BF16 conv/SiLU l2norm probe; no complete model load."""
import gc,json,statistics,time
from pathlib import Path
import torch,torch_npu
import torch.nn.functional as F
from safetensors import safe_open
from fnx import engine as E
from fnx.load import QDIR
P=Path(__file__).resolve().parent
def weight(layer,name):
 with safe_open(f'{QDIR}/L{layer:02d}.safetensors',framework='pt',device='cpu') as f:
  key=f'model.language_model.layers.{layer}.{name}.weight'
  if key in f.keys():return f.get_tensor(key).to('npu:0')
  q=f.get_tensor(key+'.q8').to('npu:0');s=f.get_tensor(key+'.s8').to('npu:0')
  pad=int(f.get_tensor(key+'.__pad')) if key+'.__pad' in f.keys() else 0
  w=E.QW(q,s,K=q.shape[-1]-pad).deq_w()
  return w.contiguous()
def compare(a,b):
 a=a.float();b=b.float();ab=a.view(torch.int32);bb=b.view(torch.int32)
 return dict(exact=bool(torch.equal(ab,bb)),different=int((ab!=bb).sum()),n=a.numel(),max_abs=float((a-b).abs().max()),rel_l2=float((a-b).norm()/a.norm().clamp_min(1e-30)))
_UNROLL={}
def graph(fn,unroll=32):
 fn();torch.npu.synchronize();g=torch.npu.NPUGraph()
 with torch.npu.graph(g):
  for _ in range(unroll):out=fn()
 _UNROLL[id(g)]=unroll
 g.replay();torch.npu.synchronize()
 return g,out
def timing(g,reps=10):
 samples=[]
 for _ in range(7):
  a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
  for _ in range(reps):g.replay()
  b.record();b.synchronize();samples.append(a.elapsed_time(b)/reps)
 return statistics.median(samples)/_UNROLL[id(g)]
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.manual_seed(144)
 result={'projection':[],'l2norm':[]}
 for layer,names in ((0,('linear_attn.in_proj_qkv','linear_attn.in_proj_z')),(0,('linear_attn.in_proj_b','linear_attn.in_proj_a')),(3,('self_attn.q_proj','self_attn.k_proj','self_attn.v_proj'))):
  ws=[weight(layer,n) for n in names];cat=torch.cat(ws,0);ns=[w.shape[0] for w in ws]
  for T in (1,128,512,2048,8192):
   x=torch.randn((T,1,E.H),device='npu:0',dtype=torch.bfloat16)*.1
   unroll=32 if T<=512 else 4
   gs,ys=graph(lambda:tuple(E.lin(x,w) for w in ws),unroll);gm,ym=graph(lambda:tuple(E.lin(x,cat).split(ns,-1)),unroll)
   torch.npu.synchronize();cmp=[compare(a,b) for a,b in zip(ys,ym)]
   times=[dict(mode=m,ms=timing(g,3 if T>=2048 else 20)) for m,g in (('split',gs),('merge',gm),('split',gs),('merge',gm))]
   row=dict(layer=layer,names=names,T=T,comparison=cmp,times=times);result['projection'].append(row);print('PROJECTION',json.dumps(row),flush=True)
   del gs,gm,ys,ym,x;gc.collect();torch.npu.empty_cache()
  del ws,cat;gc.collect();torch.npu.empty_cache()
 # Model weights, actual BF16 projection -> four-tap conv -> BF16 SiLU.
 w=weight(0,'linear_attn.in_proj_qkv');cw=weight(0,'linear_attn.conv1d').view(E.CONV_DIM,4)
 for T in (1,128,2048):
  x=torch.randn((T,E.H),device='npu:0',dtype=torch.bfloat16)*.1
  mq=F.linear(x,w);hist=torch.randn(E.CONV_DIM,3,device='npu:0',dtype=torch.bfloat16)*.1
  seq=torch.cat([hist,mq.T],-1)
  conv=torch.stack([cw[:,j:j+1]*seq[:,j:j+T] for j in range(4)],-1).sum(-1)
  y=F.silu(conv).T
  for label,t in zip(('q','k'),y.split([E.KEY_DIM,E.KEY_DIM,E.VAL_DIM],-1)):
   t=t.reshape(T,E.NKH,E.HKD)
   sq=t*t;s=sq.sum(-1,keepdim=True);den=s+1e-6;r=torch.rsqrt(den);out=t*r
   rr=torch.rsqrt(den.float()).bfloat16()
   row=dict(T=T,source=label,dtype=str(t.dtype),square_dtype=str(sq.dtype),sum_dtype=str(s.dtype),epsilon_dtype=str(den.dtype),rsqrt_dtype=str(r.dtype),fp32_square=compare(s,(t.float()*t.float()).sum(-1,keepdim=True).bfloat16()),rounded_square_fp32_sum=compare(s,sq.float().sum(-1,keepdim=True).bfloat16()),rsqrt_fp32_cast=compare(r,rr),output_fp32_rsqrt=compare(out,t*rr))
   result['l2norm'].append(row);print('BF16_NORM',json.dumps(row),flush=True)
  del x,mq,hist,seq,conv,y
 # Extremes expose each original BF16 rounding and overflow boundary.
 ex=torch.tensor([0,1e-22,1e-20,1e-10,1e-4,1,1e4,1e19,1e20],device='npu:0',dtype=torch.bfloat16)[:,None,None].expand(-1,16,128).contiguous()
 sq=ex*ex;s=sq.sum(-1,keepdim=True);den=s+1e-6;r=torch.rsqrt(den)
 result['extremes']={'values':ex[:,0,0].float().cpu().tolist(),'sums':s[:,0,0].float().cpu().tolist(),'rsqrt_fp32_cast':compare(r,torch.rsqrt(den.float()).bfloat16())}
 (P/'probe.json').write_text(json.dumps(result,indent=2));print('PROBE_COMPLETE',flush=True)
if __name__=='__main__':main()
