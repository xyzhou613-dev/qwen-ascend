import json,sys,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;phase=sys.argv[1]
h=json.loads((P/f'{phase}-health.json').read_text())
assert h['diagnostic_variant']==phase and h['worker_alive'] and h['ctx']==262144
assert h['dense_w8a8_enabled']==(phase!='rollback') and not h['kmajor_down_enabled']
req=urllib.request.Request('http://127.0.0.1:19002/__r157_audit',data=b'{}',headers={'Content-Type':'application/json'})
a=json.load(urllib.request.urlopen(req,timeout=120));layers=a['parent']['layers']+a['worker']['layers']
assert len(layers)==48 and sorted(x['layer'] for x in layers)==list(range(48))
assert len({x['pid'] for x in layers})==2
if phase!='rollback':
    q=[v for row in layers for v in row['projections'].values() if v['type']=='SmoothW8A8']
    assert len(q)==96 and all(any(t['dtype']=='torch.int8' for t in x['tensors'].values()) for x in q)
if phase=='combined':
    assert not a['parent']['heads'] and len(a['worker']['heads'])==1
    head=a['worker']['heads'][0];assert head['type']=='QuantHead' and (head['K'],head['N'])==(2560,248320)
    assert len(head['chunks'])==4 and all(c['weight_dtype']=='torch.int8' and c['scale_dtype']=='torch.float32' for c in head['chunks'])
    assert a['worker']['head_calls'].get('1',0)>0
(P/f'{phase}-runtime-audit.json').write_text(json.dumps(a,indent=2)+'\n')
startup=dict(phase=phase,http_ready_s=time.time()-int((P/f'{phase}-start.txt').read_text()),layers=48,
    smooth_projections=96 if phase!='rollback' else 0,head_a8w8_actually_captured=phase=='combined')
(P/f'{phase}-startup.json').write_text(json.dumps(startup,indent=2)+'\n');print('READY_AUDITED',startup,flush=True)
