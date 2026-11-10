import gc,json,statistics,os
from pathlib import Path
import torch,torch_npu
from fnx import engine as E
from fnx.load import load_layer,QDIR
OUT=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.manual_seed(129130)
 weights=load_layer(str(Path(QDIR)/'L00.safetensors'),0,'npu:0')
 layers={}
 for mode in ('baseline','fused'):
  os.environ['FNX_RAW_COMPACT']='1'
  layers[mode]=E.FnLayer(0,'linear',weights,'npu:0')
 print('RAW_STRIDES',{m:{n:getattr(L,n).w.stride() for n in ('a_up','m_up')} for m,L in layers.items()},flush=True)
 original=E.delta_chunk_batched_fp32;rows=[]
 for T in (1,128,512,2048):
  x=torch.randn((T,1,E.HCH),device='npu:0',dtype=torch.bfloat16)*.03
  position=torch.tensor([3],device='npu:0')
  graphs={};outputs={}
  for mode,L in layers.items():
   os.environ['FNX_CONV_PREFILL']='1';os.environ['FNX_EXPERT_GATHER']='1' if mode=='fused' else '0'
   (L.forward(x,position) if T==1 else L.forward_chunk(x,3,T));torch.npu.synchronize();g=torch.npu.NPUGraph()
   with torch.npu.graph(g):outputs[mode]=(L.forward(x,position) if T==1 else L.forward_chunk(x,3,T))
   graphs[mode]=g
  E.delta_chunk_batched_fp32=original
  acc=dict(output_exact=True,state_exact=True,conv_exact=True,output_max_abs=0,state_max_abs=0)
  for step in range(8):
   x.copy_(torch.randn_like(x)*.03)
   S=torch.randn_like(layers['baseline'].S)*.01;C=torch.randn_like(layers['baseline'].conv)*.03
   for L in layers.values():L.S.copy_(S);L.conv.copy_(C)
   for g in graphs.values():g.replay()
   torch.npu.synchronize()
   for name,a,b in (('output',outputs['baseline'],outputs['fused']),('state',layers['baseline'].S,layers['fused'].S),('conv',layers['baseline'].conv,layers['fused'].conv)):
    acc[name+'_exact'] &= torch.equal(a.view(torch.int32 if name=='state' else torch.int16),b.view(torch.int32 if name=='state' else torch.int16))
    if name!='conv':acc[name+'_max_abs']=max(acc[name+'_max_abs'],(a.float()-b.float()).abs().max().item())
  timings=[]
  for mode in ('baseline','fused','baseline','fused'):
   samples=[]
   for _ in range(5):
    a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record()
    for _ in range(8):graphs[mode].replay()
    b.record();b.synchronize();samples.append(a.elapsed_time(b)/8)
   timings.append(dict(mode=mode,ms=statistics.median(samples)))
  row=dict(T=T,accuracy=acc,timings=timings,dynamic_steps=8);rows.append(row)
  (OUT/'layer-gather.json').write_text(json.dumps(rows,indent=2));print('LAYER_CONV_PREFILL',json.dumps(row),flush=True)
  del graphs,outputs;gc.collect();torch.npu.empty_cache()
 print('LAYER_CONV_PREFILL_COMPLETE',flush=True)
if __name__=='__main__':main()
