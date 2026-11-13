import gc,json,os,statistics,sys,time,urllib.request
from pathlib import Path
import torch,torch_npu
from safetensors import safe_open
from fnx import engine as E
P=Path(__file__).resolve().parent;selector=int(sys.argv[1]);started=time.monotonic();torch.npu.set_device(0);torch.set_num_threads(1);torch.npu.set_per_process_memory_fraction(1/64,0);torch.manual_seed(1530+selector)
def guard():
 h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5));assert h['service_revision']=='r152' and h['worker_alive'];assert torch.npu.mem_get_info()[0]>1536*1024**2 and time.monotonic()-started<500

def load(layer,name):
 with safe_open(f'/models/Qwen3.8-Flash-Next-Q/L{layer:02d}.safetensors',framework='pt',device='cpu') as f:
  key=f'model.language_model.layers.{layer}.{name}'
  if key in f.keys():return f.get_tensor(key).npu()
  q=f.get_tensor(key+'.q8').npu();s=f.get_tensor(key+'.s8').npu();pad=int(f.get_tensor(key+'.__pad').item()) if key+'.__pad' in f.keys() else 0;K=q.shape[1]-pad;gs=q.shape[1]//s.shape[1]
  obj=E.DenseW8(q,s,K=K) if K%gs==0 else E.QW(q,s,K=K)
  return obj.deq_w().contiguous()

def bench(functions):
 stream=torch.npu.Stream();graphs={};outputs={};calls=64
 with torch.npu.stream(stream):
  for name,fn in functions.items():
   fn();stream.synchronize();g=torch.npu.NPUGraph()
   with torch.npu.graph(g,stream=stream):
    for _ in range(calls):out=fn()
   graphs[name]=g;outputs[name]=out;g.replay();stream.synchronize()
  times={n:[] for n in graphs}
  for repeat in range(12):
   guard()
   for n in (list(graphs) if repeat%2==0 else list(reversed(graphs))):
    a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record(stream)
    for _ in range(2):graphs[n].replay()
    b.record(stream);b.synchronize();times[n].append(a.elapsed_time(b)*1000/(2*calls));time.sleep(.01)
 for g in graphs.values():g.reset()
 return dict(times_us=times,median_us={k:statistics.median(v) for k,v in times.items()})

@torch.no_grad()
def main():
 guard();records=[];L=0 if selector==0 else 24;A=L+3
 cases=[('qkv',L,['linear_attn.in_proj_qkv.weight'],36),('z',L,['linear_attn.in_proj_z.weight'],36),('qkv_z',L,['linear_attn.in_proj_qkv.weight','linear_attn.in_proj_z.weight'],36),('out',L,['linear_attn.out_proj.weight'],48),('full_q',A,['self_attn.q_proj.weight'],12),('full_kv',A,['self_attn.k_proj.weight','self_attn.v_proj.weight'],12),('mix_up',L,['attn_hyper_connection.input_mix_weight_up.weight'],96),('mix_down_inj',L,['attn_hyper_connection.input_mix_weight_down.weight','attn_hyper_connection.block_inject_weight.weight'],96)]
 for label,layer,names,multiplicity in cases:
  guard();parts=[load(layer,n) for n in names];w=torch.cat(parts,0) if len(parts)>1 else parts[0];del parts;N,K=w.shape
  wf=w.float();ws=wf.abs().amax(1).clamp_min(1e-12)/127;qw=(wf/ws[:,None]).round().clamp(-127,127).to(torch.int8);del wf
  weights={'t':qw.t(),'nd':qw.t().contiguous()}
  if os.environ.get('PROBE_NZ')=='1':
   torch.npu.config.allow_internal_format=True;weights['nz']=torch_npu.npu_format_cast(weights['nd'],29);torch.npu.config.allow_internal_format=False
  x=torch.randn(1,K,device='npu',dtype=torch.bfloat16)*.03
  def quant(layout):
   xq,xs=torch_npu.npu_dynamic_quant(x)
   return torch_npu.npu_quant_matmul(xq,weights[layout],ws,pertoken_scale=xs,output_dtype=torch.bfloat16)
  fns={'bf16':lambda:torch.nn.functional.linear(x,w)}
  for layout in weights:fns['w8a8_'+layout]=lambda layout=layout:quant(layout)
  errors=[]
  for sample in range(16):
   x.copy_(torch.randn_like(x)*[.003,.03,.3,1.][sample%4]);a=fns['bf16']().float();b=fns['w8a8_t']().float();den=a.norm().clamp_min(1e-12)
   errors.append(dict(relative_l2=float((a-b).norm()/den),cosine=float(torch.nn.functional.cosine_similarity(a,b).mean()),max_abs=float((a-b).abs().max())))
  timing=bench(fns);row=dict(name=label,layer=layer,shape=[N,K],reference='r152 BF16 dequantized weight',weight_quant='per-output-channel symmetric int8 of current BF16 weight',activation_quant='CANN per-token dynamic int8',layouts={k:torch_npu.get_npu_format(v) for k,v in weights.items()},timing=timing,errors=errors,multiplicity_hint=multiplicity)
  records.append(row);(P/f'dense-device{selector}.json').write_text(json.dumps(dict(status='running',selector=selector,records=records),indent=2));print('DENSE_PROBE',selector,label,[N,K],timing['median_us'],flush=True)
  del fns,weights,x,w,ws,qw,a,b;gc.collect();torch.npu.empty_cache()
 (P/f'dense-device{selector}.json').write_text(json.dumps(dict(status='passed',selector=selector,records=records,max_allocated_bytes=torch.npu.max_memory_allocated(),elapsed_s=time.monotonic()-started,exclusive=False),indent=2));print('DENSE_BOTH_LAYOUTS_COMPLETE',selector,flush=True)
main()
