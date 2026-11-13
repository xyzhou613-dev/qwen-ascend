"""Checkpoint every frozen-ID case; separately label pilot, decode and true prefill."""
import argparse,hashlib,json,os,time,urllib.request
from pathlib import Path
from checkpoint import rows as checkpoint_rows,append as checkpoint_append
P=Path(__file__).resolve().parent;Q=Path('/models/fnx-diagnostics/r156-moe-kmajor-20261113/private-quality')
def post(url,route,body):
 req=urllib.request.Request(url+route,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
 return json.load(urllib.request.urlopen(req,timeout=300))
def main():
 a=argparse.ArgumentParser();a.add_argument('phase');a.add_argument('--url',default='http://127.0.0.1:19002');a.add_argument('--full',action='store_true');o=a.parse_args()
 health=json.load(urllib.request.urlopen(o.url+'/health',timeout=5));assert health['diagnostic_variant']==o.phase
 # Verify the accelerated scoring method against the legacy CPU-logit method.
 fast=post(o.url,'/__r157_score',dict(pilot=True));slow=post(o.url,'/__r157_score',dict(pilot=True,legacy=True))
 difference=max(abs(x-y) for x,y in zip(fast['nll_tokens'],slow['nll_tokens']));assert difference<.00005,(difference,fast['mean_nll'],slow['mean_nll'])
 (P/f'{o.phase}-scoring-parity.json').write_text(json.dumps(dict(fast=fast,legacy=slow,max_abs_delta=difference),indent=2))
 if o.phase in ('baseline','rollback'):assert abs(slow['mean_nll']-5.178334)<.00001
 cases=json.load(open(Q/'nll-inputs.json'))['cases'];path=P/f'{o.phase}-nll.jsonl';done={}
 if path.exists():done={r['case']:r for r in checkpoint_rows(path)}
 selected=list(range(len(cases))) if o.full else [0,49,98,147]
 for index in selected:
  if index in done:continue
  start=time.monotonic();r=post(o.url,'/__r157_score',dict(case=index));expected=cases[index]
  assert r['variant']==o.phase and r['ids_sha256']==expected['ids_sha256'] and r['scored']==512
  row=dict(case=index,domain=expected['domain'],elapsed_s=time.monotonic()-start,**r)
  checkpoint_append(path,row)
  print('NLL_CASE',o.phase,index,r['mean_nll'],round(row['elapsed_s'],2),flush=True);done[index]=row
 # Frozen r152 natural text: regression coverage, not new holdout evidence.
 fixtures=json.load(open('/models/fnx-diagnostics/r152-state-prefill-20261113/fixtures.json'));results={}
 for name in fixtures:
  r=post(o.url,'/__r157_score',dict(fixture=name,mode='prefill'));assert r['scored']==199
  results[name]=r;print('PREFILL_NLL',o.phase,name,r['mean_nll'],flush=True)
 (P/f'{o.phase}-prefill-quality.json').write_text(json.dumps(results,indent=2))
 if o.full:
  natural=json.load(open(P/'private-natural/inputs.json'));npath=P/f'{o.phase}-natural.jsonl'
  completed={r['case'] for r in checkpoint_rows(npath)}
  for index,case in enumerate(natural):
   if index in completed:continue
   r=post(o.url,'/__r157_score',dict(natural=index,mode='prefill'));assert r['ids_sha256']==case['ids_sha256'] and r['scored']==199
   checkpoint_append(npath,dict(case=index,language=case['language'],**r));print('NATURAL_PREFILL',o.phase,case['name'],r['mean_nll'],flush=True)
 print('QUALITY_NLL_COMPLETE',o.phase,len(selected),len(selected)*512,flush=True)
if __name__=='__main__':main()
