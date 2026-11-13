import hashlib,json,sys,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;phase=sys.argv[1]
ready_elapsed=time.time()-int((P/f'{phase}-start-epoch.txt').read_text())
h=json.load(open(P/f'{phase}-health.json'))
assert h['worker_alive'] and h['diagnostic_variant']==phase and h['dense_w8a8_enabled']==(phase=='candidate') and h['ctx']==262144
if phase=='candidate':
 rows=[json.loads(s) for f in P.glob('candidate-projections-*.jsonl') for s in f.read_text().splitlines()]
 assert len(rows)==48 and sorted(r['layer'] for r in rows)==list(range(48))
 assert sum(len(r['projections']) for r in rows)==96 and len({r['pid'] for r in rows})==2
 ref={r['layer']:r['sha256'] for r in json.load(open(P/'calibration-reference.json'))['layers']}
 for r in rows:
  assert ref[r['layer']]==r['calibration_sha256']
  assert all(x['type']=='SmoothW8A8' and x['weight_dtype']=='torch.int8' and x['input_scale_dtype']=='torch.bfloat16' for x in r['projections'].values())
 (P/'candidate-projection-audit.json').write_text(json.dumps(rows,indent=2))
text='A reliable inference service needs reproducible measurements. We compare the same inputs, record latency and accuracy, and check whether the result survives repeated tests.'
req=urllib.request.Request('http://127.0.0.1:19002/fnx/loss',data=json.dumps(dict(text=text,max_tokens=16)).encode(),headers={'Content-Type':'application/json'})
r=json.load(urllib.request.urlopen(req,timeout=120));(P/f'{phase}-loss16.json').write_text(json.dumps(r,indent=2))
expected=5.154368 if phase=='candidate' else 5.178334
# Startup sanity only. The candidate intentionally changes numerical execution;
# the independent r154 development-quality limits remain unchanged.
tolerance=.001 if phase=='candidate' else .00001
delta=r['mean_nll']-expected
assert abs(delta)<tolerance,(phase,r['mean_nll'],expected,tolerance)
ready=dict(phase=phase,http_ready_s=ready_elapsed,loss16=r['mean_nll'],
 reference_loss16=expected,reference_delta_nll=delta,sanity_tolerance=tolerance,
 quality_acceptance_complete=False)
(P/f'{phase}-startup.json').write_text(json.dumps(ready,indent=2));print('READY_CHECKED',ready,flush=True)
