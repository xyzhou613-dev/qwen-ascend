"""Freeze 100352 scored targets using the actual deployed tokenizer, CPU only."""
import hashlib,json
from collections import defaultdict
from pathlib import Path
from transformers import AutoTokenizer
P=Path(__file__).resolve().parent;D=P/'private-quality';old=Path('/models/fnx-diagnostics/r154-w8a8-gemv-20261113')
tok=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next')
def digest(ids):return hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
excluded=set();references=[]
for name in ('suite-inputs.json','calibration-inputs.json'):
 f=old/name
 if not f.exists():continue
 obj=json.load(open(f));cases=obj.get('cases',[]) if isinstance(obj,dict) else obj
 for row in cases:
  ids=row.get('ids',[])
  for i in range(len(ids)-127):excluded.add(digest(ids[i:i+128]))
 references.append(dict(file=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
assert len(references)==2,'missing calibration/development dedup reference'
streams=defaultdict(list);source_ids=defaultdict(list)
for line in (D/'nll-source-texts.jsonl').read_text().splitlines():
 row=json.loads(line);streams[row['domain']].extend(tok.encode(row['text']+'\n\n'));source_ids[row['domain']].append(row['id'])
cases=[]
for domain,ids in sorted(streams.items()):
 assert len(ids)>=49*577,(domain,len(ids))
 for i in range(49):
  piece=ids[i*577:(i+1)*577]
  assert all(digest(piece[j:j+128]) not in excluded for j in range(len(piece)-127)),(domain,i,'development/calibration overlap')
  cases.append(dict(name=f'{domain}/{i:02d}',domain=domain,ids=piece,warm_context=64,scored_tail=512,ids_sha256=digest(piece)))
target=D/'nll-inputs.json';assert not target.exists(),'refuse overwrite frozen NLL inputs'
target.write_text(json.dumps(dict(cases=cases,scored_targets=100352,scope='held-out benchmark knowledge/math/code; not natural-text/long-prefill/task acceptance'),ensure_ascii=False,separators=(',',':')))
manifest=dict(status='frozen_not_evaluated',scored_targets=100352,cases=196,domains={d:25088 for d in streams},
 ids_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),tokenizer_sha256=hashlib.sha256(Path('/models/Qwen3.8-Flash-Next/tokenizer.json').read_bytes()).hexdigest(),
 dedup_references=references,common_128_token_window_count=0,source_ids=source_ids,quality_acceptance_complete=False)
(P/'nll-freeze.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print('NLL_FROZEN',manifest['scored_targets'],manifest['cases'],manifest['ids_sha256'])
