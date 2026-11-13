"""Frozen regression and short NLL checks; explicitly not full acceptance."""
import hashlib,json,sys,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
R157=Path('/models/fnx-diagnostics/r157-combined-quality-20261113')
phase=sys.argv[1];assert phase in ('baseline','combined')
def post(body):
    req=urllib.request.Request('http://127.0.0.1:19002/__r157_score',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req,timeout=400))
pilot=post(dict(pilot=True));legacy=post(dict(pilot=True,legacy=True))
delta=max(abs(a-b) for a,b in zip(pilot['nll_tokens'],legacy['nll_tokens']))
assert delta<.00005
if phase=='baseline':assert abs(legacy['mean_nll']-5.178334)<.00001
parity=dict(max_abs_delta=delta,fast=pilot,legacy=legacy)
(P/f'{phase}-scoring-parity.json').write_text(json.dumps(parity,indent=2)+'\n')
fixtures=json.loads(Path('/models/fnx-diagnostics/r152-state-prefill-20261113/fixtures.json').read_text())
reference=json.loads((R157/'rollback-prefill-quality.json').read_text())
result={}
# Probe the known failure first, while still collecting all five fixed slices.
for name in sorted(fixtures,key=lambda x:(x!='natural18000',x)):
    r=post(dict(fixture=name,mode='prefill'))
    assert r['scored']==199 and r['mode']=='prefill' and r['variant']==phase
    expected=hashlib.sha256(json.dumps(fixtures[name],separators=(',',':')).encode()).hexdigest()
    assert r['ids_sha256']==expected
    if phase=='baseline':assert abs(r['mean_nll']-reference[name]['mean_nll'])<.00005,(name,r['mean_nll'],reference[name]['mean_nll'])
    result[name]=r
    (P/f'{phase}-prefill-quality.json').write_text(json.dumps(result,indent=2)+'\n')
    print('LEGACY_PREFILL',phase,name,r['mean_nll'],flush=True)
print('FIVE_REGRESSION_SLICES_COMPLETE_NOT_FULL_ACCEPTANCE',phase,flush=True)
