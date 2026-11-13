"""Paired real layers: SmoothQuant alone versus SmoothQuant + original-layout down."""
import gc,json,os,statistics,sys,time,urllib.request
from pathlib import Path
import torch,torch_npu
from fnx import engine as E
from fnx.load import load_layer,QDIR
import fnx.kmajor_down as kmajor_down
P=Path(__file__).resolve().parent;selector=int(sys.argv[1]);started=time.monotonic()
os.environ['FNX_DENSE_W8A8']='1';os.environ['FNX_KMAJOR_SO']=str(P/'libfnx_w4_kmajor.so')
os.environ['FNX_W8A8_CALIB_DIR']='/models/fnx-diagnostics/r154-w8a8-gemv-20261113/private-calibration'
torch.set_num_threads(1);torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(1/32,0);torch.manual_seed(15650+selector)
def guard():
 h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5));assert h['service_revision']=='r152' and h['worker_alive']
 assert torch.npu.mem_get_info()[0]>768*1024**2
 assert time.monotonic()-started<550
def err(a,b):
 a=a.cpu().float();b=b.cpu().float();return dict(relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),max_abs=float((a-b).abs().max()),finite=bool(torch.isfinite(b).all()))
@torch.no_grad()
def main():
 records=[]
 for index,kind in ((selector*24,'linear'),(selector*24+3,'full')):
  guard();E._GAMMA_CACHE.clear();w=load_layer(f'{QDIR}/L{index:02d}.safetensors',index,'npu:0');layers={}
  for mode in ('w8a8','combined'):
   L=E.FnLayer(index,kind,w,'npu:0')
   if kind=='full':
    L.kv=E.KVStore(8192,'npu:0',1);L.index_k=torch.zeros(8192,128,device='npu',dtype=torch.bfloat16);L.index_keys=torch.zeros(2048,128,device='npu',dtype=torch.float32)
   layers[mode]=L
  del w;torch.npu.empty_cache()
  x=torch.randn(1,1,E.HCH,device='npu',dtype=torch.bfloat16)*.03;pos=torch.tensor([4096],device='npu',dtype=torch.int64);errors=[];hits_start=kmajor_down.hits
  for sample in range(16):
   guard();x.copy_(torch.randn_like(x)*[.003,.03,.3,1.][sample%4]);result={}
   for mode,L in layers.items():
    os.environ['FNX_KMAJOR_DOWN']='1' if mode=='combined' else '0';result[mode]=L.forward(x,pos).detach().cpu()
   e=err(result['w8a8'],result['combined']);assert e['finite'] and e['relative_l2']<.03,e;errors.append(e)
  a,b=layers.values();state={'conv_equal':torch.equal(a.conv,b.conv)}
  if kind=='linear':state['fp32_state_equal']=torch.equal(a.S,b.S)
  assert all(state.values()),state
  assert kmajor_down.hits-hits_start==16,'new decode path was bypassed'
  graphs={};times={m:[] for m in layers};stream=torch.npu.Stream();torch.npu.empty_cache()
  with torch.npu.stream(stream):
   for mode,L in layers.items():
    os.environ['FNX_KMAJOR_DOWN']='1' if mode=='combined' else '0';L.forward(x,pos);stream.synchronize();g=torch.npu.NPUGraph()
    with torch.npu.graph(g,stream=stream,pool=next(iter(graphs.values())).pool() if graphs else None):
     for _ in range(16):out=L.forward(x,pos)
    graphs[mode]=g;g.replay();stream.synchronize()
   for rep in range(12):
    guard()
    for mode in (list(graphs) if rep%2==0 else list(reversed(graphs))):
     a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record(stream)
     for _ in range(2):graphs[mode].replay()
     b.record(stream);b.synchronize();times[mode].append(a.elapsed_time(b)/32)
    time.sleep(.03)
   for g in graphs.values():g.reset()
  row=dict(layer=index,kind=kind,median_ms={k:statistics.median(v) for k,v in times.items()},times_ms=times,output_errors=errors,state_checks_before_timing=state,kmajor_dispatch_hits=kmajor_down.hits)
  records.append(row);print('REAL_LAYER',index,row['median_ms'],flush=True)
  (P/f'layers-device{selector}.json').write_text(json.dumps(dict(status='running',records=records),indent=2))
  del layers,L,graphs,g,out,x,pos,result;E._GAMMA_CACHE.clear();gc.collect();torch.npu.empty_cache()
 (P/f'layers-device{selector}.json').write_text(json.dumps(dict(status='passed_component',records=records,elapsed_s=time.monotonic()-started,peak_allocated_bytes=torch.npu.max_memory_allocated(),scope='4-layer weighted proxy only; not HTTP TPOT or final quality'),indent=2))
if __name__=='__main__':main()
