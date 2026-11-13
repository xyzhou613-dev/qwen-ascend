import json,urllib.request
from pathlib import Path
p=Path(__file__).resolve().parent
text=(p/'suite-sources/eval_text.txt').read_text(encoding='utf-8-sig')
req=urllib.request.Request('http://127.0.0.1:9002/fnx/loss',data=json.dumps({'text':text,'max_tokens':320}).encode(),headers={'Content-Type':'application/json'})
r=json.load(urllib.request.urlopen(req,timeout=120));(p/'live-suite-baseline320.json').write_text(json.dumps(r,indent=2));print('LIVE_320',r['n'],r['mean_nll'],flush=True)
