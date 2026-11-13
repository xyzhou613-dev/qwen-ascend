from pathlib import Path
import copy,hashlib,json,re
from transformers import AutoTokenizer
P=Path(__file__).resolve().parent;tok=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next',trust_remote_code=True)
p=json.load(open(P/'inputs/long-payload.json'));other=json.load(open(P/'inputs/long-payload-second.json'))
def ids(p):
 text=tok.apply_chat_template(p['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
 return tok(text).input_ids
assert len(ids(p))==len(ids(other))==262016
text=p['messages'][0]['content'];m=re.search(r'[a-f0-9]{32}',text);assert m
used={re.search(r'[a-f0-9]{32}',x['messages'][0]['content']).group() for x in (p,other)}
for char in 'bcdef0123456789a':
 new=char+m.group()[1:]
 if new in used:continue
 candidate=copy.deepcopy(p);candidate['messages'][0]['content']=text[:m.start()]+new+text[m.end():];x=ids(candidate)
 if len(x)==262016:break
else:raise AssertionError('no equal-token document identifier')
(P/'inputs/long-payload-third.json').write_text(json.dumps(candidate,ensure_ascii=False))
manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'inputs').glob('*.json')};(P/'input-manifest.json').write_text(json.dumps(manifest,indent=2))
(P/'triple-long-manifest.json').write_text(json.dumps(dict(prompt_tokens=262016,max_tokens=128,three_unique_identifiers=True,third_token_ids_sha256=hashlib.sha256(json.dumps(x).encode()).hexdigest()),indent=2));print('THREE_FROZEN_262016_TOKEN_INPUTS',flush=True)
