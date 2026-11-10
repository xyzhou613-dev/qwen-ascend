from pathlib import Path
from decimal import Decimal
import csv,json,collections,statistics
P=Path(__file__).resolve().parent
runs={x['tag']:x for x in json.loads((P/'runs.json').read_text())}
def read(tag,name):return list(csv.DictReader(next((P/'traces'/tag).rglob(name)).open()))
def union(a):
 a=sorted((s,e) for s,e in a if e>s)
 if not a:return 0
 total=0;s,e=a[0]
 for x,y in a[1:]:
  if x<=e:e=max(e,y)
  else:total+=e-s;s,e=x,y
 return total+e-s
def group(n):
 if 'WeightQuant' in n:return 'W4/W8 quantized matmul'
 if 'GroupedMatmul' in n:return 'MoE down GMM'
 if 'MatMulCommon' in n or 'BatchMatMul' in n or 'MatMulV3' in n:return 'Dense + index MatMul'
 if 'InferAttention' in n or 'FlashAttention' in n:return 'Sparse selected attention'
 if 'delta_sequential' in n:return 'FP32 DeltaNet'
 if 'weighted_reduce' in n:return 'HC/MoE fused reductions'
 if 'RmsNorm' in n:return 'RMSNorm'
 if any(x in n for x in ('Gather','Transpose','Slice','Concat','TensorMove','Scatter','index_expand','dual_write')):return 'Gather/layout/cache writes'
 return 'Other vector/scalar kernels'
def window(tag):
 rows={x['step']:x for x in runs[tag]['rows']};origin=Decimal(rows[8]['realtime_ns'])/1000
 width=(rows[39]['end_ns']-rows[8]['end_ns'])/1000
 def t(s):return float(Decimal(s.strip())-origin)
 return rows,origin,width,t
out={}
for tag in ('timing','memory'):
 rows,origin,width,t=window(tag);allkernels=[];copies=[];by=collections.defaultdict(lambda:dict(count=0,us=0,read=0,write=0));stages=[]
 for st in (0,1):
  kernels=read(f'{tag}_stage{st}','kernel_details.csv');tasks=read(f'{tag}_stage{st}','task_time.csv');ks=[];ms=[]
  for r in kernels:
   s=t(r['Start Time(us)']);e=s+float(r['Duration(us)'])
   if s>=0 and e<=width:
    ks.append((s,e));allkernels.append((s,e));g=group(r['Name']);v=by[g];v['count']+=1;v['us']+=e-s
    v['read']+=sum(float(r.get(k,'0') or 0) for k in ('aic_read_main_memory_datas(KB)','aiv_read_main_memory_datas(KB)'))*1024
    v['write']+=sum(float(r.get(k,'0') or 0) for k in ('aic_write_main_memory_datas(KB)','aiv_write_main_memory_datas(KB)'))*1024
  for r in tasks:
   s=t(r['task_start(us)']);e=t(r['task_stop(us)'])
   if s>=0 and e<=width and 'MEMCPY' in r['kernel_type'].upper():ms.append((s,e));copies.append((s,e))
  stages.append(dict(stage=st,kernel_us=union(ks),copy_us=union(ms),active_us=union(ks+ms),kernel_count=len(ks),memcpy_count=len(ms)))
 active=union(allkernels+copies);ku=union(allkernels);mu=union(copies)
 per=lambda v:v/31/1000
 out[tag]=dict(period_ms=per(width),kernel_raw_sum_ms=per(sum(v['us'] for v in by.values())),kernel_union_ms=per(ku),kernel_overlap_ms=per(sum(v['us'] for v in by.values())-ku),memcpy_exclusive_ms=per(active-ku),device_idle_ms=per(width-active),stages=stages,groups={k:dict(count_per_token=v['count']/31,ms=per(v['us']),pmu_read_bytes=v['read']/31,pmu_write_bytes=v['write']/31) for k,v in by.items()})
 # Per-token spans distinguish intra-stage task gaps from cross-stage host gaps.
 intra=0;outer=0
 for i in range(9,40):
  a=(rows[i-1]['end_ns']-rows[8]['end_ns'])/1000;b=(rows[i]['end_ns']-rows[8]['end_ns'])/1000;spans=[]
  for st in (0,1):
   ints=[]
   for r in read(f'{tag}_stage{st}','task_time.csv'):
    if r['kernel_type'] in ('AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV') or 'MEMCPY' in r['kernel_type'].upper():
     s=t(r['task_start(us)']);e=t(r['task_stop(us)'])
     if s>=a and e<=b:ints.append((s,e))
   if ints:
    span=(min(s for s,e in ints),max(e for s,e in ints));spans.append(span);intra+=span[1]-span[0]-union(ints)
  outer+=b-a-union(spans)
 out[tag]['intra_stage_gap_ms']=per(intra);out[tag]['outer_host_gap_ms']=per(outer)
 print(tag,json.dumps(out[tag],ensure_ascii=False),flush=True)
(P/'time_budget.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
