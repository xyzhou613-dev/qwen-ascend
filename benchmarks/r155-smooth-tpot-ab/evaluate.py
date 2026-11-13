import json,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
data={phase:json.load(open(P/phase/'summary.json')) for phase in ('baseline','candidate','rollback')}
metrics=['median_8k_tpot_ms','median_256k_tpot_ms','cached_256k_tpot_ms','sustained1024_tpot_ms']
results={}
for key in metrics:
 a,b,c=[data[p][key] for p in ('baseline','candidate','rollback')]
 results[key]=dict(baseline=a,candidate=b,rollback=c,speedup_vs_baseline=b and a/b-1,speedup_vs_rollback=c/b-1,
     latency_reduction_vs_baseline=1-b/a,rollback_drift=c/a-1)
exact={}
for name in ('8k.json','256k.json'):
 a=json.load(open(P/'baseline'/name));b=json.load(open(P/'rollback'/name));exact[name]=all(x['answer']==y['answer'] for x,y in zip(a,b))
crosscheck={}
for phase in data:
 crosscheck[phase]={}
 for name in ('8k.json','256k.json'):
  rows=json.load(open(P/phase/name));ref=json.load(open(P/'baseline'/name))
  assert len(rows)==len(ref)==3
  assert [r['payload_sha256'] for r in rows]==[r['payload_sha256'] for r in ref]
  assert all(r['stats']['tokens']==128 for r in rows)
  crosscheck[phase][name]=[dict(server_tpot_ms=r['stats']['tpot_ms'],
   client_postfirst_ms=1000*(r['end_to_end_s']-r['ttft_s'])/(r['stats']['tokens']-1),
   n_prefill=r['stats']['n_prefill'],output_tokens=r['stats']['tokens']) for r in rows]
(P/'http-timing-crosscheck.json').write_text(json.dumps(crosscheck,indent=2))
report=dict(status='measured_candidate_not_deployed',phases=data,metrics=results,baseline_rollback_answers_exact=exact,
    quality_acceptance_complete=False,production_restored=True)
(P/'summary.json').write_text(json.dumps(report,indent=2));print('MEASURED_TPOT',json.dumps(results),flush=True)
