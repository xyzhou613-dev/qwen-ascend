"""Real full-attention layer at late-context positions; compare outputs and all cache state."""
import gc,json,os,statistics
from pathlib import Path
import torch,torch_npu
from fnx import engine as E
from fnx.load import load_layer,QDIR,SRCDIR
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.manual_seed(1433)
 cfg=json.loads((Path(SRCDIR)/'config.json').read_text())['text_config']
 index=next(i for i,t in enumerate(cfg['layer_types']) if t in E.FULL_TYPES)
 W=load_layer(str(Path(QDIR)/f'L{index:02d}.safetensors'),index,'npu:0')
 layers={};capacity=65536
 for mode in ('baseline','fused'):
  L=E.FnLayer(index,'full',W,'npu:0');L.kv=E.KVStore(capacity,'npu:0',1);L.ki=0
  L.index_k=torch.zeros((capacity,128),device='npu:0',dtype=torch.bfloat16)
  L.index_keys=torch.zeros((capacity//4,128),device='npu:0',dtype=torch.float32)
  L.index_scan_cap=capacity;layers[mode]=L
 base=layers['baseline']
 buffers=lambda L:[*L.kv.k,*L.kv.v,L.index_k,L.index_keys]
 snapshots=[]
 for t in buffers(base):
  t.normal_(std=.01);snapshots.append(t.clone())
 rows=[]
 for T in (1,128,512,2048):
  x=torch.randn((T,1,E.HCH),dtype=torch.bfloat16,device='npu:0')*.03
  position=torch.tensor([capacity-4096],dtype=torch.int64,device='npu:0')
  graphs={};outs={}
  for mode,L in layers.items():
   os.environ['FNX_INDEX_POST']='1' if mode=='fused' else '0'
   for a,b in zip(buffers(L),snapshots):a.copy_(b)
   fn=lambda: L.forward(x,position) if T==1 else L.forward_chunk(x,position,T)
   fn();torch.npu.synchronize();g=torch.npu.NPUGraph()
   with torch.npu.graph(g):outs[mode]=fn()
   graphs[mode]=g
  exact=dict(output=True,kv=True,index_raw=True,index_cached=True)
  for step in range(8):
   x.normal_(std=.03)
   for L in layers.values():
    for a,b in zip(buffers(L),snapshots):a.copy_(b)
   for g in graphs.values():g.replay()
   torch.npu.synchronize()
   exact['output'] &= torch.equal(outs['baseline'].view(torch.int16),outs['fused'].view(torch.int16))
   for j,(a,b) in enumerate(zip(buffers(layers['baseline']),buffers(layers['fused']))):
    key='kv' if j<len(snapshots)-2 else ('index_raw' if j==len(snapshots)-2 else 'index_cached')
    exact[key] &= torch.equal(a.view(torch.int32 if a.dtype==torch.float32 else torch.int16),b.view(torch.int32 if b.dtype==torch.float32 else torch.int16))
  times=[]
  for mode in ('baseline','fused','baseline','fused'):
   samples=[]
   for _ in range(5):
    a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
    for _ in range(4):graphs[mode].replay()
    b.record();b.synchronize();samples.append(a.elapsed_time(b)/4)
   times.append(dict(mode=mode,ms=statistics.median(samples)))
  row=dict(layer=index,T=T,position=capacity-4096,capacity=capacity,dynamic_steps=8,exact=exact,timings=times)
  rows.append(row);(P/'full-layer.json').write_text(json.dumps(rows,indent=2));print('REAL_FULL_LAYER',row,flush=True)
  assert all(exact.values()),row
  del graphs,outs,x,position;gc.collect();torch.npu.empty_cache()
 print('REAL_FULL_LAYER_ACCEPT',flush=True)
if __name__=='__main__':main()
