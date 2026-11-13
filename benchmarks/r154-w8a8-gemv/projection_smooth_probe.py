"""Batch-shape support and timing for the actual W8A8 projection class."""
import gc,json,statistics,sys,time,urllib.request
from pathlib import Path
import torch,torch_npu
from safetensors import safe_open
from fnx import engine as E
from smooth_w8a8 import SmoothW8A8 as W8A8
P=Path(__file__).resolve().parent;selector=int(sys.argv[1]);started=time.monotonic()
torch.set_num_threads(1);torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(1/64,0)
torch.manual_seed(15490+selector)

def guard():
 h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
 assert h['service_revision']=='r152' and h['worker_alive']
 assert torch.npu.mem_get_info()[0]>1024**3 and time.monotonic()-started<650

def load(layer,name):
 with safe_open(f'/models/Qwen3.8-Flash-Next-Q/L{layer:02d}.safetensors',framework='pt',device='cpu') as f:
  key=f'model.language_model.layers.{layer}.{name}'
  q=f.get_tensor(key+'.q8').npu();s=f.get_tensor(key+'.s8').npu()
  return E.DenseW8(q,s,K=q.shape[1]).deq_w().contiguous()

@torch.no_grad()
def main():
 records=[];L=0 if selector==0 else 24
 for label,index,names in [('qkv_z',L,['linear_attn.in_proj_qkv.weight','linear_attn.in_proj_z.weight']),
                          ('out',L,['linear_attn.out_proj.weight']),
                          ('full_q',L+3,['self_attn.q_proj.weight'])]:
  guard();parts=[load(index,name) for name in names];w=torch.cat(parts,0) if len(parts)>1 else parts[0];del parts

  with safe_open(str(P/'private-calibration'/f'L{index:02d}.safetensors'),framework='pt',device='cpu') as f:
   stats=f.get_tensor({'qkv_z':'w_qkv','out':'w_out','full_q':'w_q'}[label])
  q=W8A8.from_bf16(w,stats);K=w.shape[1];stream=torch.npu.Stream()
  assert q(torch.empty(0,1,K,device='npu',dtype=torch.bfloat16)).shape==(0,1,w.shape[0])
  for T in (1,2,4,128,512,2048):
   guard();x=torch.randn(T,1,K,device='npu',dtype=torch.bfloat16)*.03
   a=torch.nn.functional.linear(x,w).cpu().float();b=q(x).cpu().float()
   error=dict(relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),max_abs=float((a-b).abs().max()))
   assert torch.isfinite(b).all() and b.shape==a.shape
   fns={'bf16':lambda:torch.nn.functional.linear(x,w),'w8a8':lambda:q(x)}
   graphs={};outputs={};times={key:[] for key in fns};calls=64 if T<=4 else 16
   torch.npu.synchronize();torch.npu.empty_cache()
   with torch.npu.stream(stream):
    for key,fn in fns.items():
     fn();stream.synchronize();g=torch.npu.NPUGraph()
     with torch.npu.graph(g,stream=stream,pool=graphs['bf16'].pool() if graphs else None):
      for _ in range(calls):out=fn()
     graphs[key]=g;outputs[key]=out;g.replay();stream.synchronize()
    for repeat in range(8):
     guard()
     for key in (list(fns) if repeat%2==0 else list(reversed(fns))):
      aev=torch.npu.Event(enable_timing=True);bev=torch.npu.Event(enable_timing=True);aev.record(stream)
      for _ in range(2):graphs[key].replay()
      bev.record(stream);bev.synchronize();times[key].append(aev.elapsed_time(bev)*1000/(2*calls))
    # The quantization must execute for new activations on replay.
    x.copy_(torch.randn_like(x)*.3);eager=q(x);graphs['w8a8'].replay();stream.synchronize()
    assert torch.equal(eager.cpu(),outputs['w8a8'].cpu()),'dynamic replay mismatch'
   for g in graphs.values():g.reset()
   row=dict(name=label,T=T,shape=list(w.shape),error=error,times_us=times,
       median_us={k:statistics.median(v) for k,v in times.items()},dynamic_replay_exact=True,
       note='Single reused weight; hot-cache component, not an end-to-end prefill benchmark.')
   records.append(row);print('PROJECTION_RESULT',selector,label,T,row['median_us'],flush=True)
   (P/f'projection-smooth-device{selector}.json').write_text(json.dumps(dict(status='running',records=records),indent=2))
   del x,a,b,fns,graphs,outputs,g,out,eager;gc.collect();torch.npu.empty_cache()
  del w,q;gc.collect();torch.npu.empty_cache()
 (P/f'projection-smooth-device{selector}.json').write_text(json.dumps(dict(status='passed',records=records,
      exclusive=False,max_allocated_bytes=torch.npu.max_memory_allocated(),elapsed_s=time.monotonic()-started),indent=2))

if __name__=='__main__':main()
