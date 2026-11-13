import json,os,sys
from pathlib import Path
P=Path(__file__).resolve().parent;old=Path('/models/fnx-diagnostics/r155-smooth-tpot-ab-20261113')
source=(old/'http_bench.py').read_text();prefix=source[:source.index('short=[];long=[]')]
sys.argv=['smoke','recovered'];os.environ['BENCH_URL']='http://127.0.0.1:9002'
scope={'__file__':str(P/'http_bench.py')};exec(compile(prefix,'r155_sse_reader','exec'),scope)
payload=json.load(open(old/'inputs/payload1.json'));r=scope['generate'](payload)
ref=json.load(open(old/'baseline/8k.json'))[0]
assert r['stats']['tokens']==128 and r['stats']['n_prefill']>=8060
r['answer_matches_preboot_r152']=r['answer']==ref['answer'];assert r['answer_matches_preboot_r152']
r['scope']='post-reboot restoration smoke, not optimization A/B'
(P/'recovered-http.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print('RECOVERED_HTTP_PASS',r['stats']['tpot_ms'])
