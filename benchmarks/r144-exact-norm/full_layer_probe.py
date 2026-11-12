"""Real weights, stateful multi-step comparisons before complete-model tests."""
import gc,json,os,statistics
from pathlib import Path
import torch,torch_npu
from fnx import engine as E
from fnx.load import load_layer,QDIR
from probe import compare,graph,timing
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 active=[]
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit():continue
  try:
   cmd=proc.joinpath('cmdline').read_bytes().split(b'\0')
   if b'fnx.server' in cmd or b'fnx.pipeline_worker' in cmd:active.append(proc.name)
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 if active:raise SystemExit('REFUSED_LIVE_MODEL: exclusive diagnostic required; PIDs='+','.join(active))
 torch.npu.set_device(0);torch.manual_seed(1442);rows=[]
 for index,ltype in ((0,'linear'),(3,'full')):
  W=load_layer(f'{QDIR}/L{index:02d}.safetensors',index,'npu:0');layers={}
  for mode in ('baseline','norm'):
   L=E.FnLayer(index,ltype,W,'npu:0')
   if ltype=='full':
    L.kv=E.KVStore(16384,'npu:0',1);L.ki=0
    L.index_k=torch.zeros((16384,128),device='npu:0',dtype=torch.bfloat16);L.index_keys=torch.zeros((4096,128),device='npu:0',dtype=torch.float32);L.index_scan_cap=16384
   layers[mode]=L
  def buffers(L):return [L.conv]+([L.S] if ltype=='linear' else [*L.kv.k,*L.kv.v,L.index_k,L.index_keys])
  snapshots=[]
  for a in buffers(layers['baseline']):a.normal_(std=.01);snapshots.append(a.clone())
  pos=torch.tensor([8192],dtype=torch.int64,device='npu:0')
  for T in (1,128,512,2048):
   inputs=[torch.randn((T,1,E.HCH),dtype=torch.bfloat16,device='npu:0')*.03 for _ in range(4)]
   states={};times=[]
   for mode,L in layers.items():
    os.environ['FNX_NORM_POST']='0' if mode=='baseline' else '1'
    os.environ['FNX_DELTA_REUSE']='0' if mode=='baseline' else '1'
    for a,b in zip(buffers(L),snapshots):a.copy_(b)
    outputs=[]
    for x in inputs:
     y=L.forward(x,pos) if T==1 else L.forward_chunk(x,pos,T)
     outputs.append(y.clone())
    torch.npu.synchronize();states[mode]=[torch.cat(outputs,0).clone()]+[a.clone() for a in buffers(L)]
    fn=lambda: L.forward(inputs[0],pos) if T==1 else L.forward_chunk(inputs[0],pos,T)
    if os.environ.get('ACCURACY_ONLY')!='1':
     g,out=graph(fn,4)
     times.append(dict(mode=mode,ms=timing(g,4)));del g,out
   comparisons={mode:[compare(a,b) for a,b in zip(states['baseline'],states[mode])] for mode in ('norm',)}
   assert all(r['exact'] for r in comparisons['norm']),comparisons['norm']
   labels=['output','conv']+(['fp32_state'] if ltype=='linear' else [f'kv_k{i}' for i in range(len(L.kv.k))]+[f'kv_v{i}' for i in range(len(L.kv.v))]+['index_raw','index_cached'])
   row=dict(layer=index,T=T,comparison_order=labels,comparison=comparisons,times=times)
   rows.append(row);P.joinpath('full-layer-bits.json' if os.environ.get('ACCURACY_ONLY')=='1' else 'full-layer.json').write_text(json.dumps(rows,indent=2));print('FULL_LAYER',json.dumps(row),flush=True)
   del inputs,states;gc.collect();torch.npu.empty_cache()
  del layers,L,W,snapshots;gc.collect();torch.npu.empty_cache()
 print('FULL_LAYER_COMPLETE',flush=True)
if __name__=='__main__':main()
