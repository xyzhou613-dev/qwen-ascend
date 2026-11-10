from pathlib import Path
from decimal import Decimal
import json,csv,collections,math
P=Path(__file__).resolve().parent;runs={x['tag']:x for x in json.load(open(P/'runs.json'))};inv=json.load(open(P/'inventory.json'))
def shape(o):
 if o['kind']=='Raw':return o['w']['shape'][0],o['w']['shape'][1]
 if o['kind'] in ('ExpW4','DenseW8'):return o['N'],o['K']
 return o['q']['shape'][0],o['K']
def category(n):
 if n in ('a_di','a_up','m_di','m_up'):return 'HC down/up projections'
 if n in ('sh_gu','sh_down'):return 'Shared expert/router projections'
 if n in ('w_b','w_a'):return 'Delta beta/decay projections'
 if n=='index_score':return 'Indexer FP32 score'
 if n in ('w_qkv','w_z'):return 'Delta qkv/z projections'
 if n=='w_out':return 'Delta output projection'
 if n in ('w_q','w_k','w_v'):return 'QSA q/k/v projections'
 if n=='w_o':return 'QSA output projection'
 return n
out=collections.defaultdict(lambda:dict(calls=0,flops=0,timing_us=0,pmu_read=0,pmu_write=0))
byname=collections.defaultdict(lambda:dict(calls=0,timing_us=0,pmu_read=0,pmu_write=0))
for mode in ('timing','memory'):
 rr={x['step']:x for x in runs[mode]['rows']};origin=Decimal(rr[8]['realtime_ns'])/1000;width=(rr[39]['end_ns']-rr[8]['end_ns'])/1000
 for st in (0,1):
  f=next((P/'traces'/f'{mode}_stage{st}').rglob('kernel_details.csv'));rows=[]
  for row in csv.DictReader(f.open()):
   s=float(Decimal(row['Start Time(us)'].strip())-origin);e=s+float(row['Duration(us)'])
   if s>=0 and e<=width:rows.append(row)
  for row in rows:
   key=row['Name'];o=byname[key]
   if mode=='timing':o['calls']+=1;o['timing_us']+=float(row['Duration(us)'])
   else:
    o['pmu_read']+=sum(float(row[k]) for k in ('aic_read_main_memory_datas(KB)','aiv_read_main_memory_datas(KB)'))*1024
    o['pmu_write']+=sum(float(row[k]) for k in ('aic_write_main_memory_datas(KB)','aiv_write_main_memory_datas(KB)'))*1024
  specs=[];quants=[];gmms=[]
  for L in inv[f'stage{st}']['layers']:
   p=L['projections'];names=['a_di','a_up']+(['w_q','w_k','w_v','index_score','w_o'] if L['type']=='full' else ['w_qkv','w_z','w_b','w_a','w_out'])+['m_di','m_up','sh_gu','sh_down']
   for n in names:
    fl=2*4*4096*128 if n=='index_score' else 2*math.prod(shape(p[n]));specs.append((category(n),fl))
   if L['type']=='full':quants.append(('Indexer W8 projection',2*math.prod(shape(p['index_w']))))
   quants.append(('MoE gate/up W4',2*10*math.prod(shape(p['exp_gu']))));gmms.append(('MoE down W4 GMM',2*10*math.prod(shape(p['exp_dn']))))
  if st==1:
   for n in ('mix_down','mix_up'):specs.append(('LM head mixer',2*math.prod(shape(inv['stage1']['head'][n]))))
   quants += [('LM head W8',2*2560*62080)]*4
  for text,expected in (('MatMulCommon',specs),('WeightQuant',quants),('GroupedMatmul',gmms)):
   selected=[x for x in rows if text in x['Name']];assert len(selected)==31*len(expected),(mode,st,text,len(selected),len(expected))
   for i,row in enumerate(selected):
    name,flops=expected[i%len(expected)];o=out[name]
    if mode=='timing':o['calls']+=1;o['flops']+=flops;o['timing_us']+=float(row['Duration(us)'])
    else:
     o['pmu_read']+=sum(float(row[k]) for k in ('aic_read_main_memory_datas(KB)','aiv_read_main_memory_datas(KB)'))*1024
     o['pmu_write']+=sum(float(row[k]) for k in ('aic_write_main_memory_datas(KB)','aiv_write_main_memory_datas(KB)'))*1024
for data in (out,byname):
 for o in data.values():
  for k in o:o[k]/=31
(P/'semantic_matrices.json').write_text(json.dumps(out,indent=2));(P/'kernel_types.json').write_text(json.dumps(byname,indent=2))
for name,data in (('semantic_matrices',out),('kernel_types',byname)):
 rows=[dict(name=n,**o) for n,o in sorted(data.items(),key=lambda p:-p[1]['timing_us'])]
 with (P/(name+'.csv')).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=rows[0],lineterminator='\n');w.writeheader();w.writerows(rows)
print('SEMANTIC_MATMUL_TOTAL_GFLOPS',sum(x['flops'] for x in out.values())/1e9)
print(json.dumps(out,indent=2))
