import ctypes,json,statistics
from pathlib import Path
import torch,torch_npu
P=Path(__file__).resolve().parent
torch.npu.set_device(0);torch.manual_seed(141)
old=ctypes.CDLL(__import__('os').environ.get('FNX_DN_REFERENCE_SO',str(P.parents[1]/'fnx/libfnx_delta.so')));new=ctypes.CDLL(__import__('os').environ.get('FNX_DN_CANDIDATE_SO',str(P/'libtiled.so')))
old.fnx_delta_compact.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int,ctypes.c_void_p];old.fnx_delta_compact.restype=ctypes.c_int
new.fnx_delta_tiled.argtypes=[ctypes.c_void_p]*8+[ctypes.c_int]*3+[ctypes.c_void_p];new.fnx_delta_tiled.restype=ctypes.c_int
dev='npu:0';stream=torch.npu.current_stream().npu_stream
indices={r:(torch.arange(r%100,dtype=torch.int32).repeat_interleave(64 if r>=100 else 128)*4).to(dev) for r in (4,8,16,32,104,108,116,132)}
s0=torch.empty((1,48,128,128),device=dev);states={r:torch.empty_like(s0) for r in (0,4,8,16,32,104,108,116,132)}
outputs={r:torch.empty((1,48,128),device=dev) for r in states}
q=torch.empty((1,16,128),device=dev,dtype=torch.bfloat16);k=torch.empty_like(q)
v=torch.empty((1,48,128),device=dev,dtype=torch.bfloat16);beta=torch.empty((1,48),device=dev,dtype=torch.bfloat16);decay=torch.empty((1,48),device=dev)
def call(r,blocks=40):
 stream=torch.npu.current_stream().npu_stream
 args=[t.data_ptr() for t in (states[r],q,k,v,beta,decay,outputs[r])]
 rc=(old.fnx_delta_compact(*args,blocks,stream) if r==0 else new.fnx_delta_tiled(*args,indices[r].data_ptr(),r%100,64 if r>=100 else 128,blocks,stream))
 assert rc==0,rc
def inputs(step):
 q.copy_(torch.randn_like(q)*.01);k.copy_(torch.randn_like(k)*.1);v.copy_(torch.randn_like(v)*(.01 if step%2 else 10))
 beta.copy_(torch.sigmoid(torch.randn_like(beta)));decay.copy_(torch.exp(-torch.rand_like(decay)*.1))
rows=[]
with torch.no_grad():
 for step in range(128):
  inputs(step)
  if step%16==0:
   s0.normal_(std=(.01 if step%32 else 100))
   for s in states.values():s.copy_(s0)
  for r in states:call(r)
  torch.npu.synchronize()
  for r in (4,8,16,32,104,108,116,132):
   exact=torch.equal(states[0].view(torch.int32),states[r].view(torch.int32)) and torch.equal(outputs[0].view(torch.int32),outputs[r].view(torch.int32))
   rows.append(dict(kind='accuracy',step=step,rows=r,exact=exact,state_max_abs=(states[0]-states[r]).abs().max().item(),output_max_abs=(outputs[0]-outputs[r]).abs().max().item()))
 (P/'accuracy.json').write_text(json.dumps(rows,indent=2))
 assert all(r['exact'] for r in rows),[r for r in rows if not r['exact']][:5]
 print('128_DYNAMIC_ORDERED_STATE_AND_OUTPUT_EXACT',flush=True)
 if __import__('os').environ.get('PROBE_ACCURACY_ONLY')=='1':raise SystemExit(0)
 graphs={}
 for r in states:
  call(r);torch.npu.synchronize();g=torch.npu.NPUGraph()
  with torch.npu.graph(g):
   for _ in range(16):call(r)
  graphs[r]=g
 for sweep in range(3):
  for r in states:
   times=[]
   for _ in range(7):
    states[r].copy_(s0);a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record();graphs[r].replay();b.record();b.synchronize();times.append(a.elapsed_time(b)/16)
   row=dict(kind='timing',sweep=sweep,rows=r,ms=statistics.median(times));rows.append(row);print(row,flush=True)
(P/'probe.json').write_text(json.dumps(rows,indent=2))
print('PROBE_COMPLETE',flush=True)
