"""Post-only normalization: genuine conv/SiLU BF16 and denominator bit patterns."""
import ctypes,json,gc
from pathlib import Path
import torch,torch_npu
import torch.nn.functional as F
from fnx import norm_post
from probe import weight,compare,graph,timing
from fnx import engine as E
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.manual_seed(1441);rows=[]
 w=weight(0,'linear_attn.in_proj_qkv');cw=weight(0,'linear_attn.conv1d').view(E.CONV_DIM,4)
 for T in (1,128,512,2048):
  x=torch.randn(T,E.H,device='npu:0',dtype=torch.bfloat16)*.1;mq=F.linear(x,w);hist=torch.randn(E.CONV_DIM,3,device='npu:0',dtype=torch.bfloat16)*.1
  seq=torch.cat([hist,mq.T],-1);conv=torch.stack([cw[:,j:j+1]*seq[:,j:j+T] for j in range(4)],-1).sum(-1);y=F.silu(conv).T
  for label,t in zip(('q','k'),y.split([E.KEY_DIM,E.KEY_DIM,E.VAL_DIM],-1)):
   t=t.reshape(T,E.NKH,E.HKD).contiguous()
   a=E.l2norm(t);b=norm_post.run(t);torch.npu.synchronize();cmp=compare(a,b)
   ga,oa=graph(lambda:E.l2norm(t));gb,ob=graph(lambda:norm_post.run(t))
   times=[dict(mode=m,ms=timing(g,10)) for m,g in (('base',ga),('post',gb),('base',ga),('post',gb))]
   row=dict(T=T,source=label,comparison=cmp,times=times);rows.append(row);print('NORM_POST',json.dumps(row),flush=True)
   del ga,gb,oa,ob;gc.collect();torch.npu.empty_cache()
  del x,mq,hist,seq,conv,y
 # Exhaust all nonnegative finite BF16 sums (positive infinity included).
 bits=torch.arange(0,32641,dtype=torch.int32).to(torch.int16)
 bits=F.pad(bits,(0,32768-len(bits)))
 sums=bits.view(torch.bfloat16).to('npu:0').reshape(2048,16,1)
 x=torch.ones((2048,16,128),device='npu:0',dtype=torch.bfloat16);out=torch.empty_like(x)
 rc=norm_post._lib.fnx_norm_lookup(x.data_ptr(),sums.data_ptr(),norm_post.table(x.device).data_ptr(),out.data_ptr(),32768,x.stride(0),torch.npu.current_stream(x.device).npu_stream);assert rc==0
 ref=x*torch.rsqrt(sums+1e-6);torch.npu.synchronize()
 exhaustive=compare(ref,out);bad=(ref[:,:,0]!=out[:,:,0]).flatten();exhaustive['bad_sum_bits']=bits[bad.cpu()].int().tolist()[:64]
 result=dict(rows=rows,exhaustive_nonnegative_bf16_sums=exhaustive)
 P.joinpath('norm-probe.json').write_text(json.dumps(result,indent=2));print('NORM_POST_COMPLETE',json.dumps(exhaustive),flush=True)
if __name__=='__main__':main()
