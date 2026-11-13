"""Full decode down+epilogue cost, dynamic experts and live BF16 rounding."""
import ctypes,json,statistics,sys,time,urllib.request
from pathlib import Path
import torch,torch_npu
from safetensors import safe_open
from fnx import engine as E
from fnx.moe_epilogue import merge as old_merge
from fnx import kmajor_down as D
P=Path(__file__).resolve().parent;device=int(sys.argv[1]);start=time.monotonic()
torch.set_num_threads(1);torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(1/32,0);torch.manual_seed(15680+device)
def guard():
 h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5));assert h['service_revision']=='r152' and h['worker_alive']
 assert torch.npu.mem_get_info()[0]>768*2**20 and time.monotonic()-start<500
def err(a,b):
 a=a.cpu().float();b=b.cpu().float();return dict(relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),max_abs=float((a-b).abs().max()),finite=bool(torch.isfinite(b).all()))
@torch.no_grad()
def main():
 guard();cfg=json.load(open('/models/fnx-diagnostics/r155-smooth-tpot-ab-20261113/production-selected-config.json'));layer=24*device
 with safe_open(cfg['FNX_W4A16_DIR']+f'-em/L{layer:02d}.w4.safetensors',framework='pt',device='cpu') as f:
  k='mlp.experts.down_proj';obj=E.ExpW4(f.get_tensor(k+'.w4').npu(),f.get_tensor(k+'.s8t').npu(),f.get_tensor(k+'.base').npu(),64,expert_major=True)
 x=torch.randn(10,640,device='npu',dtype=torch.bfloat16)*.03;ids=torch.arange(10,device='npu',dtype=torch.int64)
 w=torch.softmax(torch.randn(1,10,device='npu'),-1).to(torch.bfloat16);sh=torch.randn(1,2560,device='npu',dtype=torch.bfloat16)*.03;gate=torch.zeros(1,1,device='npu',dtype=torch.bfloat16)
 D.apply(obj,x,ids);lib=D._lib;lib.fnx_w4_fused.argtypes=[ctypes.c_void_p]*9;lib.fnx_w4_fused.restype=ctypes.c_int
 def baseline(ix):return old_merge(obj.apply(x,ix),w,sh,gate)
 def split(ix):return D.merge(obj,x,ix,w,sh,gate)
 def fused(ix):
  y=torch.empty_like(sh);assert lib.fnx_w4_fused(obj.w4.data_ptr(),obj._scales.data_ptr(),ix.data_ptr(),x.data_ptr(),w.data_ptr(),sh.data_ptr(),gate.data_ptr(),y.data_ptr(),torch.npu.current_stream().npu_stream)==0;return y
 errors=[]
 for sample in range(16):
  guard();ids.copy_(torch.randint(0,512,(10,),device='npu'));x.copy_(torch.randn_like(x)*[.003,.03,.3,1.][sample%4]);gate.fill_([-15,-2,0,2,15][sample%5])
  if sample==0:ids.fill_(511)
  a,b,c=baseline(ids),split(ids),fused(ids);e=err(a,c);pair=err(b,c)
  assert e['finite'] and e['relative_l2']<.03 and torch.equal(b,c),(e,pair)
  errors.append(e)
 banks=torch.stack([(torch.arange(10)+i*10)%512 for i in range(64)]).npu();stream=torch.npu.Stream();graphs={};times={k:[] for k in ('baseline','split','fused')}
 with torch.npu.stream(stream):
  for name,fn in [('baseline',baseline),('split',split),('fused',fused)]:
   fn(banks[0]);stream.synchronize();g=torch.npu.NPUGraph()
   with torch.npu.graph(g,stream=stream,pool=next(iter(graphs.values())).pool() if graphs else None):
    for bank in banks:out=fn(bank)
   graphs[name]=g;g.replay();stream.synchronize()
  for rep in range(12):
   guard()
   for name in (list(graphs) if rep%2==0 else list(reversed(graphs))):
    a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record(stream)
    for _ in range(2):graphs[name].replay()
    b.record(stream);b.synchronize();times[name].append(a.elapsed_time(b)*1000/128)
   time.sleep(.03)
  for g in graphs.values():g.reset()
  for fn in (split,fused):
   g=torch.npu.NPUGraph()
   with torch.npu.graph(g,stream=stream):out=fn(ids)
   stream.synchronize()
   for _ in range(6):
    ids.copy_(torch.randint(0,512,(10,),device='npu'));x.copy_(torch.randn_like(x)*.03);eager=fn(ids);g.replay();stream.synchronize();assert torch.equal(out,eager)
   g.reset()
 result=dict(status='passed_component',device=device,layer=layer,errors=errors,fused_equals_split=True,times_us=times,median_us={k:statistics.median(v) for k,v in times.items()},peak_allocated_bytes=torch.npu.max_memory_allocated(),scope='down + production-rounded epilogue only; not HTTP')
 (P/f'merge-device{device}.json').write_text(json.dumps(result,indent=2));print('MERGE_RESULT',device,result['median_us'],flush=True)
if __name__=='__main__':main()
