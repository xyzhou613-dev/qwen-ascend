import json,sys,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
req=urllib.request.Request('http://127.0.0.1:9002/__r144_quality',data=b'{}',headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=900) as f:r=json.load(f)
(P/(sys.argv[1]+'-quality.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2))
print('MULTITEXT_PREFILL_QUALITY',sys.argv[1],{k:(v['tokens'],v['mean_nll'],v['top1']) for k,v in r.items()},flush=True)
