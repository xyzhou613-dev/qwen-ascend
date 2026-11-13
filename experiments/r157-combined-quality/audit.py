"""Startup-only metadata and graph-capture dispatch audit; no tensor copies."""
import hashlib,json,os
from pathlib import Path
import torch
rows=[];pointer_layers={};hits={};installed=False
def status():return dict(pid=os.getpid(),layers=rows,kmajor_hits=hits)
def install(E):
 global installed
 if installed:return
 installed=True
 from . import kmajor_down as D
 original_merge=D.merge
 def tracked_merge(projection,*args):
  result=original_merge(projection,*args)
  if result is not None:
   index=pointer_layers[projection.w4.data_ptr()];hits[index]=hits.get(index,0)+1
  return result
 D.merge=tracked_merge
 ctor=E.FnLayer.__init__
 def initialize(layer,*args,**kwargs):
  ctor(layer,*args,**kwargs);entry=dict(layer=layer.idx,pid=os.getpid(),visible=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'),projections={})
  for name in ('w_qkvz','w_out','w_q','w_o'):
   obj=getattr(layer,name,None)
   if obj is None:continue
   tensors={}
   for cls in type(obj).__mro__:
    for key in getattr(cls,'__slots__',()):
     value=getattr(obj,key,None)
     if isinstance(value,torch.Tensor):tensors[key]=dict(shape=list(value.shape),dtype=str(value.dtype))
   entry['projections'][name]=dict(type=type(obj).__name__,tensors=tensors)
  dn=getattr(layer,'exp_dn',None)
  if dn is not None:pointer_layers[dn.w4.data_ptr()]=layer.idx
  rows.append(entry)
  p=Path(os.environ['R157_DIR'])/'audits';p.mkdir(exist_ok=True)
  with (p/f'{os.environ["R157_PHASE"]}-{os.getpid()}.jsonl').open('a') as f:f.write(json.dumps(entry)+'\n')
 E.FnLayer.__init__=initialize
