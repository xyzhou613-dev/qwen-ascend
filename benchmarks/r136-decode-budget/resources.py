from pathlib import Path
import json,csv,collections
P=Path(__file__).resolve().parent;d=json.loads((P/'inventory.json').read_text());records=[]
def shape(o):
 k=o['kind']
 if k=='Raw':return o['w']['shape'][0],o['w']['shape'][1]
 if k in ('DenseW8','ExpW4'):return o['N'],o['K']
 if k=='QW':return o['q']['shape'][0],o['K']
 if k=='Tensor':return o['shape'][0],o['shape'][1]
def bytes_min(o):
 k=o['kind']
 if k=='Raw':return o['w']['bytes']
 if k=='DenseW8':
  if o.get('chunks'):return sum(x['weight']['bytes']+x['scale']['bytes'] for x in o['chunks'])
  return o['w8t']['bytes']+o['sc']['bytes']
 if k=='QW':return o['q']['bytes']+o['s']['bytes']
 if k=='ExpW4':return (o['w4']['bytes']+o['_scales']['bytes'])*10/o['E']
 if k=='Tensor':return o['bytes']
def category(n,layer):
 if n in ('a_di','a_up','m_di','m_up'):return 'Hyper connection projections'
 if n in ('sh_gu','sh_down'):return 'Shared expert + router'
 if n.startswith('exp_'):return 'MoE up W4' if n=='exp_gu' else 'MoE down W4'
 if n=='index_w':return 'Indexer projection W8'
 if n.startswith('w_'):return 'Linear-attn projections' if layer!='full' else 'Full-attn projections'
 return 'Head projections'
for stage in ('stage0','stage1'):
 for L in d[stage]['layers']:
  for n,o in L['projections'].items():
   N,K=shape(o);mult=10 if o['kind']=='ExpW4' else 1
   records.append(dict(stage=stage,layer=L['index'],name=n,group=category(n,L['type']),kind=o['kind'],N=N,K=K,mult=mult,flops=2*N*K*mult,weight_min_bytes=bytes_min(o)))
for n,o in d['stage1']['head'].items():
 N,K=shape(o);records.append(dict(stage='stage1',layer='head',name=n,group='LM head W8' if n=='lm_head' else 'Head mixer QW',kind=o['kind'],N=N,K=K,mult=1,flops=2*N*K,weight_min_bytes=bytes_min(o)))
groups=collections.defaultdict(lambda:dict(flops=0,weight_min_bytes=0,projections=0))
for x in records:
 for f in ('flops','weight_min_bytes'):groups[x['group']][f]+=x[f]
 groups[x['group']]['projections']+=1
extra=dict(index_score_flops=12*2*4*128*4096,attention_valid_flops=12*4*24*256*2049.5,attention_width2051_flops=12*4*24*256*2051,delta_flops=36*5505024,delta_native_executed_flops=36*5517312,delta_state_read_write_bytes=36*2*48*128*128*4,hidden_D2H_H2D_bytes=2*10240*2)
result=dict(groups=dict(groups),matrix_flops=sum(x['flops'] for x in records),weight_min_bytes=sum(x['weight_min_bytes'] for x in records),extra=extra,records=records)
(P/'resources.json').write_text(json.dumps(result,indent=2))
with (P/'projection_flops_bytes.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=records[0],lineterminator='\n');w.writeheader();w.writerows(records)
print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
