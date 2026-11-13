"""Freeze held-out task inputs and NLL source texts before candidate evaluation.

Public benchmarks are held out from FNX calibration/development; this does not
assert that the original pretrained model never saw these public benchmarks.
Raw dataset copies remain private; commit manifests and preparation code only.
"""
import csv,gzip,hashlib,io,json,tarfile
from collections import Counter,defaultdict
from pathlib import Path
P=Path(__file__).resolve().parent;S=P/'quality-sources';D=P/'private-quality';D.mkdir(exist_ok=True)
def sha(b):return hashlib.sha256(b).hexdigest()
def rank(key):return sha(('fnx-holdout-156-v1:'+str(key)).encode())
def sample(rows,n):return sorted(rows,key=lambda x:rank(x['id']))[:n]
def stratified(rows,n):
 groups=defaultdict(list)
 for r in rows:groups[r['slice']].append(r)
 buckets=[sorted(v,key=lambda x:rank(x['id'])) for k,v in sorted(groups.items())];result=[]
 while len(result)<n:
  changed=False
  for bucket in buckets:
   if bucket and len(result)<n:result.append(bucket.pop(0));changed=True
  assert changed,'insufficient source examples'
 return result
def write(name,rows):
 path=D/name;assert not path.exists(),f'refuse overwrite frozen {name}'
 path.write_bytes(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in rows).encode('utf-8'))
 return dict(file=name,sha256=sha(path.read_bytes()),rows=len(rows),bytes=path.stat().st_size)
manifest=json.loads((S/'source-manifest.json').read_text(encoding='utf-8-sig'))
for source in manifest['sources']:assert sha((S/source['file']).read_bytes())==source['sha256']
gsm=[json.loads(s) for s in (S/'gsm8k-test.jsonl').read_text(encoding='utf-8').splitlines()]
mbpp=[json.loads(s) for s in (S/'mbpp.jsonl').read_text(encoding='utf-8').splitlines()]
hum=[json.loads(s) for s in gzip.decompress((S/'HumanEval.jsonl.gz').read_bytes()).decode().splitlines()]
cm=[]
with tarfile.open(S/'cmmlu.tar.gz') as tf:
 for member in tf.getmembers():
  if '/data/test/' not in member.name or not member.name.endswith('.csv'):continue
  subject=Path(member.name).stem
  for i,row in enumerate(csv.DictReader(io.StringIO(tf.extractfile(member).read().decode('utf-8-sig')))):
   cm.append(dict(id=f'cmmlu/{subject}/{i}',slice=subject,prompt=row['Question']+'\n'+'\n'.join(k+'. '+row[k] for k in 'ABCD')+'\n请只回答正确选项字母。',
    expected=row['Answer'],kind='mcq',max_tokens=32,reference_text=row['Question']+'\n'+'\n'.join(k+'. '+row[k] for k in 'ABCD')+'\n答案：'+row['Answer']+'. '+row[row['Answer']]))
mm=[];seen=set()
for f in sorted(S.glob('mmlu-pro-*.json')):
 for item in json.loads(f.read_text(encoding='utf-8'))['rows']:
  row=item['row'];key=str(row['question_id'])
  if key in seen:continue
  seen.add(key);letters='ABCDEFGHIJ'
  mm.append(dict(id='mmlu-pro/'+key,slice=row['category'],kind='mcq',
   prompt=row['question']+'\n'+'\n'.join(letters[i]+'. '+v for i,v in enumerate(row['options']))+'\nReturn only the letter of the correct option.',expected=row['answer'],max_tokens=32,
   reference_text=row['question']+'\n'+'\n'.join(letters[i]+'. '+v for i,v in enumerate(row['options']))+'\nAnswer: '+row['answer']+'. '+row['options'][row['answer_index']]))
gm=[dict(id=f'gsm8k/test/{i}',slice='math',kind='numeric',prompt=r['question']+'\nExplain briefly and write the final answer as #### NUMBER.',expected=r['answer'].rsplit('####',1)[1].strip(),max_tokens=768,reference_text=r['question']+'\n'+r['answer']) for i,r in enumerate(gsm)]
he=[dict(id='humaneval/'+r['task_id'],slice='python',kind='code',prompt='Complete the following Python function. Return only the complete Python code, including the function signature.\n'+r['prompt'],expected=None,entry_point=r['entry_point'],tests=r['test'],canonical_code=r['prompt']+r['canonical_solution'],max_tokens=1024,reference_text=r['prompt']+r['canonical_solution']) for r in hum]
mp=[dict(id=f'mbpp/{r["task_id"]}',slice='python',kind='code',prompt='Write Python code for this task. Use the function names shown in the tests. Return only Python code.\n'+r['text']+'\n'+'\n'.join(r['test_list']),expected=None,tests=r.get('test_setup_code','')+'\n'+'\n'.join(r['test_list']),canonical_code=r['code'],max_tokens=1024,reference_text=r['text']+'\n'+r['code']) for r in mbpp if 11<=r['task_id']<=510]
tasks=stratified(cm,300)+stratified(mm,300)+sample(gm,200)+sample(he,100)+sample(mp,100)
# Exact JSON data-extraction/argument tasks are a separate synthetic slice,
# not a claim to have exercised the service's native tool-call protocol.
for i in range(100):
 records=[dict(id=f'R{i:03d}-{j}',value=((i+11)*(j+3))%97,active=(i+j)%3!=0) for j in range(7)]
 expected=dict(request_id=f'Q{i:03d}',selected=sorted(r['id'] for r in records if r['active'] and r['value']>=40),total=sum(r['value'] for r in records if r['active']))
 tasks.append(dict(id=f'structure/{i}',slice='json_extract',kind='json',prompt='读取记录，仅输出一个JSON对象，不要Markdown。字段request_id照抄；selected为active=true且value>=40的id，按字典序排序；total为所有active=true记录的value之和。\n'+json.dumps(dict(request_id=f'Q{i:03d}',records=records),ensure_ascii=False),expected=expected,max_tokens=192))
assert len(tasks)==1100 and len({t['id'] for t in tasks})==1100
task_rows=[{k:v for k,v in t.items() if k!='reference_text'} for t in sorted(tasks,key=lambda x:rank(x['id']))]
nll=[]
for domain,pool in [('zh_knowledge',cm),('en_knowledge',mm),('math',gm),('python',he+mp)]:
 for row in sample(pool,min(600,len(pool))):
  nll.append(dict(id=row['id'],domain=domain,text=row['reference_text']))
files=[write('tasks.jsonl',task_rows),write('nll-source-texts.jsonl',nll)]
result=dict(status='frozen_inputs_not_evaluated',tasks=1100,task_counts=dict(Counter(t['kind'] for t in tasks)),
 slices={'cmmlu':dict(Counter(t['slice'] for t in tasks if t['id'].startswith('cmmlu/'))),'mmlu_pro':dict(Counter(t['slice'] for t in tasks if t['id'].startswith('mmlu-pro/')))},
 files=files,source_manifest_sha256=sha((S/'source-manifest.json').read_bytes()),
 nll_scope='Benchmark knowledge/math/code text; natural-text and long-context acceptance still required. Actual-model tokenization and >=100k scored-target freeze pending.',
 quality_acceptance_complete=False,pretraining_contamination='unknown; no claim of unseen pretraining data',
 candidate_tuning_on_holdout=False,seed='fnx-holdout-156-v1',thresholds=dict(mean_delta_nll=.02,slice_delta_nll=.05,task_noninferiority_margin=.02))
(P/'quality-freeze.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8');print('QUALITY_FROZEN',result['tasks'],result['task_counts'],files)
