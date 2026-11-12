from html.parser import HTMLParser
from pathlib import Path
import hashlib,json
from transformers import AutoTokenizer
P=Path(__file__).resolve().parent
class Extract(HTMLParser):
 def __init__(self):super().__init__();self.stack=[];self.parts=[];self.buf=None
 def handle_starttag(self,tag,attrs):
  a=dict(attrs);self.stack.append((tag,a))
  if tag=='p' and any('mw-parser-output' in x.get('class','').split() for _,x in self.stack):self.buf=[]
 def handle_endtag(self,tag):
  if tag=='p' and self.buf is not None:
   s=''.join(self.buf).strip()
   if s:self.parts.append(s)
   self.buf=None
  for i in range(len(self.stack)-1,-1,-1):
   if self.stack[i][0]==tag:del self.stack[i:];break
 def handle_data(self,data):
  if self.buf is not None and not any(t in ('style','script','sup') for t,a in self.stack):self.buf.append(data)
T=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next',trust_remote_code=True)
records=[];texts=[]
urls=['https://zh.wikisource.org/zh-hans/阿Ｑ正傳','https://zh.wikisource.org/wiki/祝福','https://zh.wikisource.org/wiki/弟兄','https://zh.wikisource.org/wiki/離婚','https://zh.wikisource.org/wiki/幸福的家庭','https://zh.wikisource.org/wiki/傷逝_(魯迅)','https://zh.wikisource.org/wiki/孤獨者','https://zh.wikisource.org/wiki/在酒樓上']
for name,url in zip(('aq-page','zhufu','dixiong','lihun','xingfu','shangshi','guduzhe','jiulou'),urls):
 raw=(P/'corpus-html'/f'{name}.html').read_bytes();e=Extract();e.feed(raw.decode());text='\n'.join(e.parts);assert len(text)>3000,(name,len(text))
 ids=T.encode(text,add_special_tokens=False);print('CORPUS',name,len(text),len(ids),text[:45],flush=True)
 texts.append(ids);records.append(dict(file=name+'.html',url=url,html_sha256=hashlib.sha256(raw).hexdigest(),text_sha256=hashlib.sha256(text.encode()).hexdigest(),characters=len(text),tokens=len(ids),paragraphs=len(e.parts)))
allids=[i for seq in texts for i in seq];assert len(allids)>=54000,len(allids)
f=json.load(open(P/'fixtures.json'));assert len(f)==3
# One pass over unique sources: disjoint slices, no repeated padding.
f['natural18000']=allids[:18000];f['natural36000']=allids[18000:54000]
(P/'fixtures.json').write_text(json.dumps(f))
m=dict(sources=records,author='鲁迅',method='mw-parser-output paragraph extraction, concatenate sources once in manifest order; disjoint 0:18000 and 18000:54000 token slices; no repeated padding',fixtures={k:dict(tokens=len(v),token_ids_sha256=hashlib.sha256(json.dumps(v,separators=(',',':')).encode()).hexdigest()) for k,v in f.items()})
(P/'quality-input-manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2));print('FROZEN_FIXTURES',[(k,len(v)) for k,v in f.items()],flush=True)
