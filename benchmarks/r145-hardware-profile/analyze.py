"""Closed union timeline, separate PMU passes and system HBM/PCIe samples."""
import collections,csv,json,math,statistics
from decimal import Decimal
from pathlib import Path
P=Path(__file__).resolve().parent
DEVICE={'AI_CORE','AI_VECTOR_CORE','MIX_AIC','MIX_AIV'}
def number(v):
 try:
  x=float(str(v).strip());return x if math.isfinite(x) else None
 except (ValueError,TypeError):return None
def ns(v):return int(Decimal(str(v).strip())*1000)
def union(rows):
 rows=sorted((a,b) for a,b in rows if b>a)
 if not rows:return 0
 total=0;a,b=rows[0]
 for x,y in rows[1:]:
  if x<=b:b=max(b,y)
  else:total+=b-a;a,b=x,y
 return total+b-a
def cut(rows,a,b):return [(max(x,a),min(y,b)) for x,y in rows if y>a and x<b]
def group(name):
 n=name.lower()
 if 'weightquant' in n:return 'quantized_matmul'
 if 'matmul' in n:return 'dense_index_matmul'
 if 'norm_lookup' in n:return 'norm_lookup'
 if 'delta_' in n:return 'fp32_deltanet'
 if 'gather_pair' in n:return 'expert_gather'
 if 'attention' in n:return 'sparse_attention'
 if 'rmsnorm' in n:return 'rmsnorm'
 if any(x in n for x in ('index_post','score_groups','post_scale')):return 'index_post'
 if any(x in n for x in ('gather','transpose','concat','scatter','indexselect','tensor')):return 'layout_cache'
 return 'other_vector_native'
def csvs(tag,stage,name):
 fs=list((P/'traces'/f'{tag}_stage{stage}').rglob(name));assert len(fs)==1,(tag,stage,name,fs)
 return list(csv.DictReader(fs[0].open()))
def kernels(tag,stage):
 rows=csvs(tag,stage,'kernel_details.csv')
 for r in rows:r['_start']=ns(r['Start Time(us)']);r['_end']=r['_start']+ns(r['Duration(us)'])
 return rows
def taskrows(tag,stage):
 out=[]
 for r in csvs(tag,stage,'task_time.csv'):
  typ=r['kernel_type'].upper()
  if typ in DEVICE or 'MEMCPY' in typ:out.append((ns(r['task_start(us)']),ns(r['task_stop(us)']),typ))
 return out
def bounds(record):return record['rows'][0]['end_ns'],record['rows'][-1]['end_ns'],len(record['rows'])-1
def timing(record):
 tag=record['tag'];a,b,n=bounds(record);tasks={s:taskrows(tag,s) for s in (0,1)};kr={s:kernels(tag,s) for s in (0,1)}
 dev=intra=outer=overlap=0;stage_busy=[0,0];stage_span=[0,0];handoff=[];before=[];after=[];groups=collections.defaultdict(lambda:dict(calls=0,time_ns=0))
 periods=[]
 for left,right in zip(record['rows'],record['rows'][1:]):
  lo,hi=left['end_ns'],right['end_ns'];pieces=[];spans=[];per=[]
  for stage in (0,1):
   q=cut([(x,y) for x,y,_ in tasks[stage]],lo,hi);per.append(q);pieces+=q;stage_busy[stage]+=union(q)
   if q:
    span=(min(x for x,y in q),max(y for x,y in q));spans.append(span);stage_span[stage]+=span[1]-span[0]
  busy=union(pieces);sp=union(spans);dev+=busy;intra+=sp-busy;outer+=(hi-lo)-sp;overlap+=sum(union(q) for q in per)-busy
  if len(spans)==2:
   handoff.append(spans[1][0]-spans[0][1]);before.append(spans[0][0]-lo);after.append(hi-spans[1][1])
  periods.append(dict(seq=right['seq'],width_ms=(hi-lo)/1e6,device_ms=busy/1e6,intra_ms=(sp-busy)/1e6,outer_ms=((hi-lo)-sp)/1e6))
 for stage in (0,1):
  for r in kr[stage]:
   if r['_start']>=a and r['_end']<=b:
    g=groups[group(r['Name'])];g['calls']+=1;g['time_ns']+=r['_end']-r['_start']
 result=dict(periods=n,host_period_ms=(b-a)/n/1e6,device_busy_ms=dev/n/1e6,intra_stage_gap_ms=intra/n/1e6,outer_gap_ms=outer/n/1e6,cross_device_overlap_ms=overlap/n/1e6,
  stage_device_ms=[x/n/1e6 for x in stage_busy],stage_span_ms=[x/n/1e6 for x in stage_span],handoff_gap_ms=statistics.mean(handoff)/1e6,pre_stage0_gap_ms=statistics.mean(before)/1e6,post_stage1_gap_ms=statistics.mean(after)/1e6,
  groups={k:dict(calls_per_token=v['calls']/n,ms_per_token=v['time_ns']/n/1e6) for k,v in groups.items()},per_token=periods)
 result['closure_ms']=result['device_busy_ms']+result['intra_stage_gap_ms']+result['outer_gap_ms']
 assert abs(result['closure_ms']-result['host_period_ms'])<1e-8
 # Merge measured CPU spans and device tasks using the same epoch origin.
 origin=record['rows'][0]['start_ns'];events=[]
 for stage in (0,1):
  events.append(dict(ph='M',name='process_name',pid=stage,args={'name':f'Stage{stage} Host'}))
  events.append(dict(ph='M',name='process_name',pid=10+stage,args={'name':f'NPU{stage} kernels'}))
  for e in json.load(open(P/f'{tag}_host{stage}.json')):
   events.append(dict(ph='X',name=e['name'],pid=stage,tid=0,ts=(e['start_ns']-origin)/1000,dur=(e['end_ns']-e['start_ns'])/1000,args={'seq':e['seq']}))
  for r in kr[stage]:
   events.append(dict(ph='X',name=r['Name'],pid=10+stage,tid=r['Stream ID'],ts=(r['_start']-origin)/1000,dur=(r['_end']-r['_start'])/1000,args={'task':r['Task ID'],'core':r['Accelerator Core']}))
 for r in record['rows']:
  events.append(dict(ph='X',name=f"token {r['seq']} pos {r['position']}",pid=20,tid=0,ts=(r['start_ns']-origin)/1000,dur=(r['end_ns']-r['start_ns'])/1000))
 (P/'aligned').mkdir(exist_ok=True);(P/'aligned'/f'{tag}.json').write_text(json.dumps(dict(traceEvents=events,displayTimeUnit='ms',epoch_origin_ns=origin)))
 return result
def pmu(record):
 a,b,n=bounds(record);result={}
 for stage in (0,1):
  rows=[r for r in kernels(record['tag'],stage) if r['_start']>=a and r['_end']<=b];fields=[k for k in rows[0] if k.startswith(('aic_','aiv_')) or k=='cube_utilization(%)'];groups=collections.defaultdict(list)
  for r in rows:groups[group(r['Name'])].append(r)
  def stats(sub):
   out=dict(rows=len(sub),counter_rows=sum(any(number(r.get(k)) not in (None,0) for k in fields) for r in sub),fields={})
   for key in fields:
    valid=[(r,number(r.get(key))) for r in sub if number(r.get(key)) is not None]
    if not valid:continue
    if key.endswith('(KB)') or 'cache_hit' in key or 'miss_allocate' in key or key.endswith('_total_cycles'):
     out['fields'][key]=dict(total=sum(v for _,v in valid),per_token=sum(v for _,v in valid)/n,valid_rows=len(valid))
    else:
     weightkey='aic_total_cycles' if key.startswith('aic_') else 'aiv_total_cycles' if key.startswith('aiv_') else 'Duration(us)'
     den=sum(number(r.get(weightkey)) or 0 for r,v in valid)
     out['fields'][key]=dict(weighted_mean=sum((number(r.get(weightkey)) or 0)*v for r,v in valid)/den if den else None,valid_rows=len(valid),weight=weightkey)
   hit=sum(number(r.get(k)) or 0 for r in sub for k in fields if 'read_cache_hit' in k)
   miss=sum(number(r.get(k)) or 0 for r in sub for k in fields if 'read_cache_miss_allocate' in k)
   if hit+miss:out['l2_read_hit_rate']=hit/(hit+miss);out['l2_read_hits']=hit;out['l2_read_misses']=miss
   return out
  result[str(stage)]=dict(total=stats(rows),groups={g:stats(sub) for g,sub in groups.items()})
 return result
def hardware(record):
 a=record['rows'][0]['start_ns'];b=record['rows'][-1]['end_ns'];width=(b-a)/1e9
 files=list((P/'system'/record['context']).rglob('msprof_*.json'));assert files,record['tag']
 data=json.load(open(files[-1]));events=data if isinstance(data,list) else data['traceEvents'];names={};labels={}
 for e in events:
  if e.get('ph')=='M':
   if e['name']=='process_name':names[e['pid']]=e.get('args',{}).get('name')
   if e['name']=='process_labels':labels[e['pid']]=e.get('args',{}).get('labels')
 channels=collections.defaultdict(list)
 for e in events:
  if e.get('ph')=='C' and names.get(e['pid']) in ('HBM','PCIe'):
   for key,val in e.get('args',{}).items():
    if number(val) is not None:channels[(labels[e['pid']],names[e['pid']],e['name'],key)].append((ns(e['ts']),number(val)))
 out={};samples=[]
 for (device,kind,name,key),points in channels.items():
  points.sort();integral=covered=0;inside=[]
  for i,(ts,value) in enumerate(points[:-1]):
   lo=max(ts,a);hi=min(points[i+1][0],b)
   if hi>lo:integral+=value*(hi-lo)/1e9;covered+=(hi-lo)/1e9
   if a<=ts<=b:inside.append(value)
  out.setdefault(device,dict(hbm_read_MB=0.,hbm_write_MB=0.,hbm_read_MB_s=0.,hbm_write_MB_s=0.,channels=[]))
  row=dict(kind=kind,name=name,key=key,window_mean=integral/width,integrated_MB=integral if kind=='HBM' else None,samples=len(inside),window_coverage=covered/width,peak=max(inside) if inside else None)
  out[device]['channels'].append(row)
  if kind=='HBM':
   direction='read' if 'Read' in key else 'write';out[device][f'hbm_{direction}_MB']+=integral;out[device][f'hbm_{direction}_MB_s']+=integral/width
  samples.extend(dict(device=device,kind=kind,name=name,key=key,ts_ns=t,value=v) for t,v in points if a<=t<=b)
 (P/'aligned'/f"{record['context']}_system_samples.json").write_text(json.dumps(samples))
 return dict(window_seconds=width,steps=len(record['rows']),devices=out)
def main():
 records=json.load(open(P/'runs.json'));summary=dict(revision='r144',sample_id='r145',timing={},pmu={},hardware={},baselines={},clock_alignment=[])
 for r in records:
  if r['metric']=='AiCoreNone':summary['timing'][r['context']]=timing(r)
  elif r['metric']:summary['pmu'].setdefault(r['context'],{})[r['metric']]=dict(profiled_mean_ms=r['mean_ms'],stages=pmu(r))
  elif r['tag'].endswith('hardware'):summary['hardware'][r['context']]=hardware(r)
  else:summary['baselines'][r['tag']]=dict(mean_ms=r['mean_ms'],median_ms=r['median_ms'],steps=r['steps'])
 for rec in json.load(open(P/'clock-alignment.json')):
  best=min(rec['samples'],key=lambda r:r['rtt_ns']);summary['clock_alignment'].append(dict(context=rec['context'],best_rtt_us=best['rtt_ns']/1000,child_epoch_offset_from_midpoint_us=(best['child_ns']-(best['parent_start_ns']+best['parent_end_ns'])/2)/1000,mono_offset_us=(best['child_mono_ns']-best['parent_mono_mid_ns'])/1000))
 summary['notes']=['Timing and PMU sessions are distinct; counter-session latency is not normal TPOT.','MemoryAccess fields are preserved as CANN GM/main-memory counters, not assumed equal to physical HBM. System HBM counters are collected separately.','MAC/Vector ratios are instruction cycle occupancy, not achieved peak FLOPS. CPU waits include NPU execution and must not be added twice.']
 (P/'summary.json').write_text(json.dumps(summary,indent=2));print('ANALYSIS_COMPLETE')
 for c,r in summary['timing'].items():print(c,{k:v for k,v in r.items() if k not in ('groups','per_token')})
 print('CLOCK',summary['clock_alignment'])
if __name__=='__main__':main()
