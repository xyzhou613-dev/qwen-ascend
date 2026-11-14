import json
from pathlib import Path
P=Path(__file__).resolve().parent
policy=json.loads((P/'run-policy.json').read_text())
a=json.loads((P/'rollback/summary.json').read_text());b=json.loads((P/'smooth/summary.json').read_text());c=json.loads((P/'combined/summary.json').read_text())
metrics={}
for key in ('median_8k_tpot_ms','median_256k_tpot_ms'):
    gain=a[key]/c[key]-1;saving=b[key]-c[key]
    metrics[key]=dict(output_speed_gain=gain,head_incremental_saved_ms=saving)
old=json.loads((P/'rollback-prefill-quality.json').read_text());new=json.loads((P/'combined-prefill-quality.json').read_text())
assert old.keys()==new.keys()
deltas={k:new[k]['mean_nll']-old[k]['mean_nll'] for k in old}
passed=all(x['output_speed_gain']>=policy['minimum_output_speed_gain_fraction'] and x['head_incremental_saved_ms']>=policy['head_incremental_min_saved_ms'] for x in metrics.values()) and all(d<=policy['short_prefill_slice_nll_limit'] for d in deltas.values())
r=dict(passed=passed,performance=metrics,legacy_prefill_deltas=deltas,full_quality_acceptance=False)
(P/'short-gate.json').write_text(json.dumps(r,indent=2)+'\n');print('SHORT_GATE',r,flush=True)
assert passed,'SHORT_GATE_FAILED_RESTORE_R152'
