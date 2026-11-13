"""Resumable frozen-task evaluation; code executes in CPU-only isolated containers."""
import argparse,decimal,hashlib,json,re,subprocess,time,urllib.request,uuid
from pathlib import Path
P=Path(__file__).resolve().parent
JUDGE_IMAGE='quay.io/ascend/vllm-ascend:v0.23.0'
def code_text(text):
 m=re.fullmatch(r'\s*```(?:python)?\s*\n(.*?)\n```\s*',text,re.S)
 return m.group(1) if m else text.strip()
def judge_code(task,text):
 code=code_text(text)+'\n'+task['tests']
 if task.get('entry_point'):code+='\ncheck('+task['entry_point']+')\n'
 name='fnx-r156-judge-'+uuid.uuid4().hex[:12]
 runner='import contextlib,io,json,signal,sys; signal.alarm(5); s=json.load(sys.stdin); exec(compile(s,"<generated>","exec"),{})'
 cmd=['docker','run','--rm','--name',name,'--network','none','--read-only','--cpus=1','--memory=256m','--pids-limit=64','--cap-drop=ALL','--security-opt=no-new-privileges','--user=65534:65534','--workdir=/tmp','--tmpfs','/tmp:rw,nosuid,size=64m','-i','--entrypoint','python3',JUDGE_IMAGE,'-I','-c',runner]
 try:
  r=subprocess.run(cmd,input=json.dumps(code).encode(),stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=15)
  return dict(correct=r.returncode==0,judge='isolated_cpu_container',returncode=r.returncode)
 except subprocess.TimeoutExpired:return dict(correct=False,judge='isolated_cpu_container',error='timeout')
 finally:subprocess.run(['docker','rm','-f',name],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
def grade(task,text):
 kind=task['kind']
 if kind=='code':return judge_code(task,text)
 if kind=='mcq':
  m=re.fullmatch(r'\s*(?:Answer\s*[:：]\s*)?([A-J])[.)]?\s*',text,re.I)
  got=m.group(1).upper() if m else None
 elif kind=='numeric':
  m=re.search(r'####\s*([-+\d.,]+)\s*$',text)
  try:got=str(decimal.Decimal(m.group(1).replace(',','')).normalize()) if m else None
  except decimal.InvalidOperation:got=None
  return dict(correct=got is not None and decimal.Decimal(got)==decimal.Decimal(task['expected'].replace(',','')),parsed=got)
 else:
  try:got=json.loads(text)
  except json.JSONDecodeError:got=None
 return dict(correct=got==task['expected'],parsed=got)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--phase',required=True);ap.add_argument('--url',default='http://127.0.0.1:9002');ap.add_argument('--kinds',default='mcq,numeric,code,json');ap.add_argument('--limit',type=int,default=1100);ap.add_argument('--self-test',action='store_true');a=ap.parse_args()
 assert re.fullmatch('[a-zA-Z0-9_-]+',a.phase)
 tasks=[json.loads(s) for s in (P/'private-quality/tasks.jsonl').read_text().splitlines()]
 source_sha=hashlib.sha256((P/'private-quality/tasks.jsonl').read_bytes()).hexdigest()
 if a.self_test:
  tests=[]
  for kind in ('mcq','numeric','json','code'):
   row=next(t for t in tasks if t['kind']==kind)
   expected=row['canonical_code'] if kind=='code' else json.dumps(row['expected']) if kind=='json' else '#### '+row['expected'] if kind=='numeric' else row['expected']
   result=grade(row,expected);assert result['correct'],(row['id'],result);tests.append(dict(kind=kind,id=row['id'],passed=True))
  negative=judge_code(dict(tests='assert f(1)==2'), 'def f(x): return 0');assert not negative['correct']
  tests.append(dict(kind='incorrect_code_rejected',passed=True))
  (P/'quality-harness-selftest.json').write_text(json.dumps(tests,indent=2));print('QUALITY_HARNESS_SELFTEST_PASS');return
 kinds=set(a.kinds.split(','));selected=[t for t in tasks if t['kind'] in kinds][:a.limit]
 output=P/f'quality-{a.phase}.jsonl';done={}
 if output.exists():
  done={r['id']:r for r in (json.loads(s) for s in output.read_text().splitlines())}
  assert all(r['input_sha256']==source_sha and r['url']==a.url for r in done.values())
 for task in selected:
  if task['id'] in done:continue
  payload=dict(messages=[dict(role='user',content=task['prompt'])],temperature=0,max_tokens=task['max_tokens'],enable_thinking=False,stream=False)
  req=urllib.request.Request(a.url+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
  start=time.perf_counter();response=json.load(urllib.request.urlopen(req,timeout=180));choice=response['choices'][0];text=choice['message']['content'];result=grade(task,text)
  row=dict(id=task['id'],kind=task['kind'],slice=task['slice'],answer=text,result=result,usage=response['usage'],finish_reason=choice.get('finish_reason'),elapsed_s=time.perf_counter()-start,input_sha256=source_sha,url=a.url)
  with output.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
  done[task['id']]=row;print('TASK',a.phase,task['id'],result['correct'],flush=True);time.sleep(.2)
 summary=dict(status='partial_screen' if len(selected)<1100 else 'task_execution_complete',selected=len(selected),completed=sum(t['id'] in done for t in selected),correct=sum(done[t['id']]['result']['correct'] for t in selected if t['id'] in done),input_sha256=source_sha,quality_acceptance_complete=False)
 (P/f'quality-{a.phase}-summary.json').write_text(json.dumps(summary,indent=2));print(summary)
if __name__=='__main__':main()
