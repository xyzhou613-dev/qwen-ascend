import json,sys,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;phase=sys.argv[1];h=json.load(open(P/f'{phase}-health.json'))
assert h['diagnostic_variant']==phase and h['worker_alive'] and h['ctx']==262144
assert h['dense_w8a8_enabled']==(phase in ('smooth','combined')) and h['kmajor_down_enabled']==(phase=='combined')
req=urllib.request.Request('http://127.0.0.1:19002/__r157_audit',data=b'{}',headers={'Content-Type':'application/json'})
audit=json.load(urllib.request.urlopen(req,timeout=120));rows=audit['parent']['layers']+audit['worker']['layers']
assert len(rows)==48 and sorted(r['layer'] for r in rows)==list(range(48))
assert len({r['pid'] for r in rows})==2
if phase in ('smooth','combined'):
 objects=[o for r in rows for o in r['projections'].values() if o['type']=='SmoothW8A8'];assert len(objects)==96
 assert all(any(t['dtype']=='torch.int8' and len(t['shape'])==2 for t in o['tensors'].values()) for o in objects)
if phase=='combined':
 hits={int(k):v for side in ('parent','worker') for k,v in audit[side]['kmajor_hits'].items()}
 assert sorted(hits)==list(range(48)) and min(hits.values())>0,hits
(P/f'{phase}-runtime-audit.json').write_text(json.dumps(audit,indent=2))
startup=dict(phase=phase,http_ready_s=time.time()-int((P/f'{phase}-start.txt').read_text()),layers=48,
 smooth_projections=96 if phase in ('smooth','combined') else 0,kmajor_all_layers_captured=phase=='combined')
(P/f'{phase}-startup.json').write_text(json.dumps(startup,indent=2));print('READY_CHECKED',startup,flush=True)
