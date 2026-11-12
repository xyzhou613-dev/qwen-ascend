import copy,hashlib,json,re,shutil
from pathlib import Path
from transformers import AutoTokenizer
P=Path(__file__).resolve().parent;OLD=Path('/models/fnx-diagnostics/r150-ordered-norm-20261112')
shutil.copytree(OLD/'inputs',P/'inputs',dirs_exist_ok=True)
tok=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next',trust_remote_code=True)
payload=json.load(open(P/'inputs/long-payload.json'))
def ids(p):return tok.encode(tok.apply_chat_template(p['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False),add_special_tokens=False)
a=ids(payload);assert len(a)==262016,len(a)
text=payload['messages'][0]['content'];match=re.search(r'[a-f0-9]{32}',text);assert match
old=match.group();chosen=None
for letter in 'abcdef0123456789':
 if letter==old[0]:continue
 candidate=copy.deepcopy(payload);candidate['messages'][0]['content']=text[:match.start()]+letter+old[1:]+text[match.end():]
 b=ids(candidate)
 if len(b)==len(a):chosen=candidate;break
assert chosen is not None
(P/'inputs/long-payload-second.json').write_text(json.dumps(chosen,ensure_ascii=False))
m={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/'inputs').glob('*.json')}
(P/'input-manifest.json').write_text(json.dumps(m,indent=2));(P/'long-repeat-manifest.json').write_text(json.dumps(dict(prompt_tokens=len(a),max_tokens=128,different_document_identifier=True,first_changed_token=next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y),original_token_ids_sha256=hashlib.sha256(json.dumps(a).encode()).hexdigest(),second_token_ids_sha256=hashlib.sha256(json.dumps(b).encode()).hexdigest()),indent=2));print('TWO_FROZEN_LONG_INPUTS',len(a),len(b),flush=True)
