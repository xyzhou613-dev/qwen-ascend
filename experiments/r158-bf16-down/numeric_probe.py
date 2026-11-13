"""One real down weight, <=1/32 allocator; numerical checks only, no Engine."""
import ctypes,hashlib,json,os,statistics,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
assert int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))>4*1024**2
os.environ['FNX_W4_SCALE_CACHE']='1';os.environ['FNX_EXPERT_GATHER']='1';os.environ['FNX_DECODE_DN_GMM']='0'
import torch,torch_npu
from safetensors import safe_open
from fnx import engine as E
torch.set_num_threads(1);torch.manual_seed(15820261113)
torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(1/32,0)
started=time.monotonic()

def guard():
 h=json.load(urllib.request.urlopen('http://127.0.0.1:19002/health',timeout=5))
 assert h.get('diagnostic_variant')=='rollback' and h['worker_alive']
 assert torch.npu.mem_get_info()[0]>2*1024**3
 assert time.monotonic()-started<160

def compare(a,b):
 a=a.cpu().float();b=b.cpu().float()
 return dict(relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),
             max_abs=float((a-b).abs().max()),exact_fraction=float((a==b).float().mean()),
             finite=bool(torch.isfinite(b).all()))

def load(path,symbol):
 lib=ctypes.CDLL(str(path));fn=getattr(lib,symbol);fn.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p];fn.restype=ctypes.c_int
 return lib,fn

@torch.no_grad()
def main():
 guard()
 cfg=json.load(open('/models/fnx-diagnostics/r155-smooth-tpot-ab-20261113/production-selected-config.json'))
 src=Path(cfg['FNX_W4A16_DIR']+'-em')/'L00.w4.safetensors'
 with safe_open(str(src),framework='pt',device='cpu') as f:
  key='mlp.experts.down_proj';w=f.get_tensor(key+'.w4');s=f.get_tensor(key+'.s8t');base=f.get_tensor(key+'.base')
 obj=E.ExpW4(w.npu(),s.npu(),base.npu(),64,expert_major=True)
 assert (obj.K,obj.N,obj.E)==(640,2560,512) and obj._scales.shape==(512,10,2560)
 oldlib,old=load(Path('/models/fnx-diagnostics/r156-moe-kmajor-20261113/libfnx_w4_kmajor.so'),'fnx_w4_kmajor')
 newlib,new=load(P/'libfnx_w4_bf16w.so','fnx_w4_bf16w')
 x=torch.empty((10,640),device='npu',dtype=torch.bfloat16);ids=torch.empty(10,device='npu',dtype=torch.int64)
 def call(fn,tile):
  out=torch.empty((10,2560),device='npu',dtype=torch.bfloat16)
  rc=fn(obj.w4.data_ptr(),obj._scales.data_ptr(),ids.data_ptr(),x.data_ptr(),out.data_ptr(),tile,torch.npu.current_stream().npu_stream)
  assert rc==0,rc
  return out
 records=[];covered=set()
 for sample in range(53):
  guard();ix=(torch.arange(10)+sample*10)%512
  if sample==52:ix.fill_(511)
  covered.update(ix.tolist());ids.copy_(ix.npu());x.copy_(torch.randn_like(x)*[.003,.03,.3,1.][sample%4])
  ref=obj.apply_diag(x,ids);r=dict(sample=sample,old=compare(ref,call(old,256)),rounded={})
  for tile in (64,128,256):
   v=compare(ref,call(new,tile));assert v['finite'];r['rounded'][str(tile)]=v
  records.append(r)
 assert len(covered)==512
 # Independent formula explicitly rounds each dequantized weight to BF16.
 xc=x.cpu().float();outs=[]
 for slot,e in enumerate(ids.cpu().tolist()):
  q=w[e];nib=torch.stack([(q>>k)&15 for k in range(0,32,4)],-1).reshape(640,2560)
  val=((nib+8)%16-8).float();sc=(s[:,e].to(torch.bfloat16)*base[e]).float()
  rounded=(val.reshape(10,64,2560)*sc[:,None,:]).to(torch.bfloat16).float()
  outs.append((rounded*xc[slot].reshape(10,64,1)).sum(1).sum(0))
 independent=compare(torch.stack(outs).to(torch.bfloat16),call(new,256));assert independent['relative_l2']<.001,independent
 stream=torch.npu.Stream();fresh={}
 for tile in (128,256):
  with torch.npu.stream(stream):
   call(new,tile);stream.synchronize();g=torch.npu.NPUGraph()
   with torch.npu.graph(g,stream=stream):captured=call(new,tile)
   stream.synchronize()
   for _ in range(8):
    guard();ids.copy_(torch.randint(0,512,(10,),device='npu'));x.copy_(torch.randn_like(x)*.03)
    eager=call(new,tile);g.replay();stream.synchronize();assert torch.equal(eager.cpu(),captured.cpu())
   fresh[str(tile)]=True;g.reset()
 summary=dict(scope='component numerical comparison during shared use; no timing or full-model quality claim',
  layer=0,physical_device=3,allocator_fraction=1/32,experts_covered=len(covered),samples=53,
  old_max_relative_l2=max(r['old']['relative_l2'] for r in records),
  rounded_max_relative_l2={str(t):max(r['rounded'][str(t)]['relative_l2'] for r in records) for t in (64,128,256)},
  rounded_min_exact_fraction={str(t):min(r['rounded'][str(t)]['exact_fraction'] for r in records) for t in (64,128,256)},
  independent_bf16_weight_formula=independent,fresh_input_graph_verified=fresh,
  peak_allocated_bytes=torch.npu.max_memory_allocated(),elapsed_s=time.monotonic()-started,
  code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'bf16_weight_down.cpp',P/'libfnx_w4_bf16w.so')},records=records)
 (P/'numeric-result.json').write_text(json.dumps(summary,indent=2)+'\n')
 print('BF16_WEIGHT_NUMERIC_RESULT',{k:v for k,v in summary.items() if k!='records'},flush=True);guard()

if __name__=='__main__':main()
