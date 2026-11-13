import json
from pathlib import Path
P=Path(__file__).resolve().parent
b=json.load(open(P/'baseline/summary.json'));c=json.load(open(P/'candidate/summary.json'))
qb=json.load(open(P/'baseline-quality.json'));qc=json.load(open(P/'candidate-quality.json'))
assert set(qb)==set(qc)==set(json.load(open(P/'fixtures.json'))) and sum(v['scored'] for v in qb.values())==995
quality={k:dict(input_equal=v['token_ids_sha256']==qc[k]['token_ids_sha256'],nll_exact=v['nll_tokens']==qc[k]['nll_tokens'],argmax_exact=v['argmax']==qc[k]['argmax']) for k,v in qb.items()}
a=json.load(open(P/'baseline/8k.json'));d=json.load(open(P/'candidate/8k.json'));assert len(a)==len(d)==3
answers8=[x['answer']==y['answer'] for x,y in zip(a,d)]
answerslong={f:json.load(open(P/'baseline'/f))['answer']==json.load(open(P/'candidate'/f))['answer'] for f in ('256k.json','256k-second.json','256k-third.json','cached256k.json')}
reductions=[100*(1-y/x) for x,y in zip(b['ttft_256k_runs_s'],c['ttft_256k_runs_s'])];assert len(reductions)==3
reduction=100*(1-c['ttft_256k_s']/b['ttft_256k_s'])
nonregress=dict(ttft8=c['median_8k_ttft_s']<=b['median_8k_ttft_s']*1.01,tpot8=c['median_8k_tpot_ms']<=b['median_8k_tpot_ms']*1.01,tpot256=c['tpot_256k_ms']<=b['tpot_256k_ms']*1.01)
accepted=all(all(v.values()) for v in quality.values()) and all(answers8) and all(answerslong.values()) and reduction>.5 and all(x>0 for x in reductions) and all(nonregress.values())
r=dict(baseline=b,candidate=c,quality=quality,answers_8k_exact=answers8,answers_256k_and_cached_exact=answerslong,ttft_256k_reduction_pct=reduction,individual_long_reductions_pct=reductions,nonregression=nonregress,accepted=accepted,threshold='995 targets and all answers exact; median long TTFT >0.5% lower, all three long runs lower; 8k TTFT and both TPOT regress <=1%')
(P/'ab-summary.json').write_text(json.dumps(r,indent=2));print('RESTART_AB_RESULT',json.dumps(r),flush=True)
