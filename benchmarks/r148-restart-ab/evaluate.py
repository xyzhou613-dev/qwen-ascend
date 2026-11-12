import json,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
b=json.load(open(P/'baseline/summary.json'));c=json.load(open(P/'candidate/summary.json'))
qb=json.load(open(P/'baseline-quality.json'));qc=json.load(open(P/'candidate-quality.json'))
quality={name:dict(input_equal=qb[name]['token_ids_sha256']==qc[name]['token_ids_sha256'],nll_exact=qb[name]['nll_tokens']==qc[name]['nll_tokens'],argmax_exact=qb[name]['argmax']==qc[name]['argmax']) for name in qb}
ab=json.load(open(P/'baseline/8k.json'));ac=json.load(open(P/'candidate/8k.json'))
answers8=[x['answer']==y['answer'] for x,y in zip(ab,ac)]
answers256=json.load(open(P/'baseline/256k.json'))['answer']==json.load(open(P/'candidate/256k.json'))['answer']
speed8=b['median_8k_tpot_ms']/c['median_8k_tpot_ms']-1;speed256=b['tpot_256k_ms']/c['tpot_256k_ms']-1
accepted=all(all(row.values()) for row in quality.values()) and all(answers8) and answers256 and speed8>.005 and speed256>.005
r=dict(baseline=b,candidate=c,quality=quality,answers_8k_exact=answers8,answer_256k_exact=answers256,speedup_8k_pct=100*speed8,speedup_256k_pct=100*speed256,accepted=accepted,threshold='quality/answers exact and >0.5% output speed gain at both contexts')
(P/'ab-summary.json').write_text(json.dumps(r,indent=2));print('RESTART_AB_RESULT',json.dumps(r),flush=True)
