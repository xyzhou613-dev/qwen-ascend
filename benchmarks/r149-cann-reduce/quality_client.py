import json,os,sys,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;phase=sys.argv[1]
base=os.environ.get('BENCH_URL','http://127.0.0.1:19002')
request=urllib.request.Request(base+'/__r149_quality',data=b'{}',headers={'Content-Type':'application/json'})
with urllib.request.urlopen(request,timeout=900) as response:r=json.load(response)
(P/f'{phase}-quality.json').write_text(json.dumps(r,ensure_ascii=False,indent=2))
print('REAL_PREFILL_QUALITY',phase,{k:(v['tokens'],v['mean_nll'],v['top1']) for k,v in r.items()},flush=True)
