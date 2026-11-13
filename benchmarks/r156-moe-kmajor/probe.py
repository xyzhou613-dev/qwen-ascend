"""Bounded real-weight down projection probe; never constructs an Engine."""
import ctypes,gc,hashlib,json,os,statistics,sys,time,urllib.request
from pathlib import Path
import torch,torch_npu
from safetensors import safe_open
from fnx import engine as E
P=Path(__file__).resolve().parent;device=int(sys.argv[1]);started=time.monotonic()
torch.set_num_threads(1);torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(1/32,0)
torch.manual_seed(15600+device)
lib=ctypes.CDLL(str(P/'libfnx_w4_kmajor.so'));lib.fnx_w4_kmajor.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p];lib.fnx_w4_kmajor.restype=ctypes.c_int
def guard():
 h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
 assert h['service_revision']=='r152' and h['worker_alive']
 assert torch.npu.mem_get_info()[0]>768*1024**2
 assert time.monotonic()-started<500
def compare(a,b):
 a=a.cpu().float();b=b.cpu().float();return dict(relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),max_abs=float((a-b).abs().max()),finite=bool(torch.isfinite(b).all()))
@torch.no_grad()
def main():
 guard();cfg=json.load(open('/models/fnx-diagnostics/r155-smooth-tpot-ab-20261113/production-selected-config.json'))
 layer=int(os.environ.get('PROBE_LAYER','0'));src=Path(cfg['FNX_W4A16_DIR']+'-em')/f'L{layer:02d}.w4.safetensors'
 with safe_open(str(src),framework='pt',device='cpu') as f:
  key='mlp.experts.down_proj';w=f.get_tensor(key+'.w4');s=f.get_tensor(key+'.s8t');base=f.get_tensor(key+'.base')
  obj=E.ExpW4(w.npu(),s.npu(),base.npu(),64,expert_major=True)
 assert obj.K==640 and obj.N==2560 and obj.E==512 and obj._scales.shape==(512,10,2560)
 x=torch.randn(10,640,device='npu',dtype=torch.bfloat16)*.03;ids=torch.arange(10,device='npu',dtype=torch.int64)
 def baseline(ix):return obj.apply_diag(x,ix)
 def custom(ix,tile):
  out=torch.empty(10,2560,device='npu',dtype=torch.bfloat16)
  assert lib.fnx_w4_kmajor(obj.w4.data_ptr(),obj._scales.data_ptr(),ix.data_ptr(),x.data_ptr(),out.data_ptr(),tile,torch.npu.current_stream().npu_stream)==0
  return out
 errors={str(t):[] for t in (64,128,256)}
 for sample in range(12):
  guard();ids.copy_(torch.randint(0,512,(10,),device='npu'));x.copy_(torch.randn_like(x)*[.003,.03,.3,1.][sample%4])
  if sample==0:ids.fill_(511)
  ref=baseline(ids)
  for tile in (64,128,256):
   got=custom(ids,tile);e=compare(ref,got);errors[str(tile)].append(e)
   assert e['finite'] and e['relative_l2']<.03,(tile,e)
 # Independent CPU FP32 formula, no production Cartesian GEMM or native kernel.
 ix=ids.cpu();xc=x.cpu().float();outs=[]
 for slot,e in enumerate(ix.tolist()):
  q=w[e];nib=torch.stack([(q>>k)&15 for k in range(0,32,4)],-1).reshape(640,2560)
  val=((nib+8)%16-8).float();sc=(s[:,e].to(torch.bfloat16)*base[e]).float()
  outs.append(((val*xc[slot,:,None]).reshape(10,64,2560).sum(1)*sc).sum(0))
 cpu=torch.stack(outs);independent=compare(cpu,custom(ids,128));assert independent['relative_l2']<.01,independent
 banks=torch.stack([torch.remainder(torch.arange(10)+i*10,512) for i in range(64)]).npu()
 stream=torch.npu.Stream();graphs={};times={k:[] for k in ('baseline','tile64','tile128','tile256')};fresh={}
 torch.npu.empty_cache()
 with torch.npu.stream(stream):
  for label in times:
   fn=baseline if label=='baseline' else lambda ix,t=int(label[4:]):custom(ix,t)
   fn(banks[0]);stream.synchronize();g=torch.npu.NPUGraph()
   with torch.npu.graph(g,stream=stream,pool=next(iter(graphs.values())).pool() if graphs else None):
    for bank in banks:out=fn(bank)
   graphs[label]=g;g.replay();stream.synchronize()
  for rep in range(12):
   guard()
   for label in (list(times) if rep%2==0 else list(reversed(times))):
    a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record(stream)
    for _ in range(2):graphs[label].replay()
    b.record(stream);b.synchronize();times[label].append(a.elapsed_time(b)*1000/128)
   time.sleep(.03)
  for g in graphs.values():g.reset()
  for tile in (64,128,256):
   g=torch.npu.NPUGraph()
   with torch.npu.graph(g,stream=stream):captured=custom(ids,tile)
   stream.synchronize()
   for _ in range(6):
    ids.copy_(torch.randint(0,512,(10,),device='npu'));x.copy_(torch.randn_like(x)*.03)
    eager=custom(ids,tile);g.replay();stream.synchronize();assert torch.equal(eager.cpu(),captured.cpu())
   fresh[str(tile)]=True;g.reset()
 result=dict(status='passed_component',device=device,layer=layer,weight_layout='production E,K,N/8 int32',experts=512,K=640,N=2560,
  errors=errors,independent_cpu=independent,times_us=times,median_us={k:statistics.median(v) for k,v in times.items()},
  fresh_input_graph_verified=fresh,weight_conversion_per_request=False,extra_weight_copy=False,
  peak_allocated_bytes=torch.npu.max_memory_allocated(),elapsed_s=time.monotonic()-started,scope='down projection only, nonexclusive; not HTTP TPOT or final quality')
 (P/f'down-device{device}-L{layer:02d}.json').write_text(json.dumps(result,indent=2));print('DOWN_RESULT',device,layer,result['median_us'],flush=True)
 guard()
if __name__=='__main__':main()
