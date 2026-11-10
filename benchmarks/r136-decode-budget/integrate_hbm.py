import sqlite3,json,collections
from pathlib import Path
P=Path(__file__).resolve().parent
r=json.loads((P/'sys_request/stream.json').read_text(encoding='utf-8'));events=r['events'];i,j=8,247;a,b=[events[k]['realtime_ns'] for k in (i,j)]
assert len(events)==256
out=dict(window_start_event=i,window_end_event=j,periods=j-i,start_ns=a,end_ns=b,duration_s=(b-a)/1e9,tpot_ms=(b-a)/1e6/(j-i),bandwidth_internal_db_unit='MiB/s (2**20 B/s); verified installed hbm_model.py SCALE=32/2**20',devices=[])
for db in sorted(P.glob('sys_hbm/PROF_*/device_*/sqlite/hbm.db')):
 con=sqlite3.connect(str(db));rows=con.execute('select device_id,timestamp,bandwidth,hbmid,event_type from HBMbwData order by timestamp').fetchall();groups=collections.defaultdict(list)
 for device,t,rate,hid,kind in rows:groups[(device,hid,kind)].append((t,rate))
 d=dict(device=int(db.parts[-3].split('_')[-1]),channels=[],read_bytes=0.,write_bytes=0.)
 for (device,hid,kind),v in sorted(groups.items()):
  total=0.;covered=0.
  for (prev,_),(t,rate) in zip(v,v[1:]):
   span=max(0,min(t,b)-max(prev,a))
   if span:total+=span/1e9*rate*2**20;covered+=span
  assert abs(covered-(b-a))<1000,(db,hid,kind,covered,b-a)
  d['channels'].append(dict(hbmid=hid,event=kind,bytes=total,avg_GBps=total/((b-a)/1e9)/1e9));d[kind+'_bytes']+=total
 d['bytes_per_token']=(d['read_bytes']+d['write_bytes'])/(j-i);d['average_read_write_GBps']=(d['read_bytes']+d['write_bytes'])/((b-a)/1e9)/1e9;out['devices'].append(d)
out['total_bytes_per_token']=sum(d['bytes_per_token'] for d in out['devices']);out['total_read_bytes_per_token']=sum(d['read_bytes'] for d in out['devices'])/(j-i);out['total_write_bytes_per_token']=sum(d['write_bytes'] for d in out['devices'])/(j-i);out['aggregate_GBps']=sum(d['average_read_write_GBps'] for d in out['devices'])
(P/'hbm_integrated.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
