"""One new long-position response with 1024 output tokens reserved.

Reuse the established HTTP measurement and fact construction, but omit its
repeat/continuation so this is a bounded sustained-generation stability case.
"""
import json,os
from pathlib import Path
r=Path('/root/ascend-dev')
source=(r/'r106/r52_long_client.py').read_text()
assert source.count('target=P-128')==1
source=source.replace('target=P-128',"target=P-int(os.environ['LONG_OUTPUT'])")
source=source.replace('连续写不少于800字','连续写不少于5000字')
assert "payload['stream']=False" in source
source=source.split("payload['stream']=False")[0]
os.environ.update(TEST_CTX='262144',LONG_OUTPUT='1024',LONG_EXPLAIN='1',RESULT_DIR=str(r/'r109/sustained'))
(r/'r109/sustained').mkdir(exist_ok=True)
exec(compile(source,'sustained-http-client','exec'),{})
result=json.loads((r/'r109/sustained/result_262144.json').read_text())
assert result['prompt_tokens']==261120 and result['stats']['n_prefill']==261116
assert result['recall_hits']==3 and result['stats']['tokens']>=512
assert result['stats']['tpot_ms']<1000/15
print('R109_SUSTAINED_ACCEPT',result['stats']['tokens'],flush=True)
