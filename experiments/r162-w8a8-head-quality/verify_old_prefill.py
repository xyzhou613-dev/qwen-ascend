import json
from pathlib import Path
P=Path(__file__).resolve().parent
old=json.loads(Path('/models/fnx-diagnostics/r157-combined-quality-20261113/rollback-prefill-quality.json').read_text())
new=json.loads((P/'rollback-prefill-quality.json').read_text());assert old.keys()==new.keys()
delta={k:abs(old[k]['mean_nll']-new[k]['mean_nll']) for k in old}
assert max(delta.values())<.00005,delta
(P/'baseline-prefill-repeat.json').write_text(json.dumps(delta,indent=2)+'\n')
print('BASELINE_FIVE_PREFILL_FIXTURES_REPRODUCED',flush=True)
