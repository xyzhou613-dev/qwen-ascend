import json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parent
# Reuse the SSE implementation without running the full benchmark module.
s=(P/'http_bench.py').read_text();prefix=s[:s.index('short=[];long=[]')]
sys.argv=['http_bench.py','formal'];os.environ['BENCH_URL']='http://127.0.0.1:9002'
scope={'__file__':str(P/'http_bench.py')};exec(compile(prefix,str(P/'http_bench.py'),'exec'),scope)
payload=json.load(open(P/'inputs/payload1.json'));r=scope['generate'](payload)
assert r['stats']['tokens']==128 and r['stats']['n_prefill']>=8060
ref_path=P/'baseline/8k.json'
if not ref_path.exists():ref_path=Path('/models/fnx-diagnostics/r157-combined-quality-20261113/baseline/8k.json')
ref=json.load(open(ref_path))[0]
r['answer_matches_baseline']=r['answer']==ref['answer'];assert r['answer_matches_baseline']
(P/'production-http.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print('FORMAL_R152_HTTP_PASS',r['stats']['tpot_ms'],flush=True)
