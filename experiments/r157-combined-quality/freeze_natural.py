import hashlib,json,os,re
from pathlib import Path
os.environ['TOKENIZERS_PARALLELISM']='false'
os.environ['USE_TORCH']='0'
from transformers import AutoTokenizer
P=Path(__file__).resolve().parent
assert not (P/'combined-start.txt').exists(),'freeze before candidate execution'
tok=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next');cases=[];sources=[]
references=[];old_windows=set()
root=Path('/models/fnx-diagnostics/r154-w8a8-gemv-20261113')
for name in ('suite-inputs.json','calibration-inputs.json'):
 f=root/name;obj=json.load(open(f));assert obj['cases']
 for row in obj['cases']:
  seq=row['ids']
  for j in range(len(seq)-127):old_windows.add(tuple(seq[j:j+128]))
 references.append(dict(file=name,sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
mask=(1<<64)-1;base=1000003;power=pow(base,127,1<<64)
def rolling(seq):
 h=0
 for i,v in enumerate(seq):
  if i>=128:h=(h-(seq[i-128]+1)*power)&mask
  h=(h*base+v+1)&mask
  if i>=127:yield i-127,h
old_hashes={next(rolling(w))[1] for w in old_windows}
for language,number in [('en',2600),('zh',24264)]:
 path=P/'private-natural'/f'pg{number}.txt';text=path.read_text(encoding='utf-8-sig')
 start=re.search(r'\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*',text,re.I)
 end=re.search(r'\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK',text,re.I)
 assert start and end and end.start()>start.end()
 body=text[start.end():end.start()];ids=tok.encode(body,add_special_tokens=False);assert len(ids)>=266112,(language,len(ids))
 for start_index,fingerprint in rolling(ids):
  if fingerprint in old_hashes:assert tuple(ids[start_index:start_index+128]) not in old_windows,'development/calibration overlap'
 sources.append(dict(language=language,ebook=number,url=f'https://www.gutenberg.org/cache/epub/{number}/pg{number}.txt',sha256=hashlib.sha256(path.read_bytes()).hexdigest(),available_tokens=len(ids)))
 for length in (8192,32768,65536,131072,262016):
  part=ids[4096:4096+length];cases.append(dict(name=f'{language}/{length}',language=language,ids=part,scored_tail=199,ids_sha256=hashlib.sha256(json.dumps(part,separators=(',',':')).encode()).hexdigest()))
out=P/'private-natural/inputs.json';assert not out.exists();out.write_text(json.dumps(cases,separators=(',',':')))
manifest=dict(status='frozen_not_evaluated',cases=10,scored_targets=1990,input_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),sources=sources,dedup_references=references,common_128_token_windows=0,
 scope='fresh bilingual literary natural text with true prefill at five lengths; not unseen-pretraining claim or comprehensive long-QA accuracy')
(P/'natural-freeze.json').write_text(json.dumps(manifest,indent=2));print('NATURAL_PREFILL_FROZEN',len(cases),manifest['input_sha256'])
