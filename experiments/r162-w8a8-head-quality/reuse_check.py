"""Repeat baseline domain sentinels before using old immutable checkpoints."""
import hashlib,json,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
old={r['case']:r for r in map(json.loads,(P/'rollback-nll.jsonl').read_text().splitlines())}
assert len(old)==196
result=[]
for index in (0,49,98,147):
    req=urllib.request.Request('http://127.0.0.1:19002/__r157_score',data=json.dumps(dict(case=index)).encode(),headers={'Content-Type':'application/json'})
    r=json.load(urllib.request.urlopen(req,timeout=300));ref=old[index]
    assert r['variant']=='rollback' and r['ids_sha256']==ref['ids_sha256'] and r['scored']==512
    delta=max(abs(a-b) for a,b in zip(r['nll_tokens'],ref['nll_tokens']))
    assert delta<.00005,(index,delta)
    result.append(dict(case=index,max_abs_delta=delta,ids_sha256=r['ids_sha256']))
(P/'baseline-reuse-repeat.json').write_text(json.dumps(result,indent=2)+'\n')
print('BASELINE_REUSE_DOMAIN_SENTINELS_VERIFIED',flush=True)
