import json
from pathlib import Path
P=Path(__file__).resolve().parent
rows=[r for d in (0,1) for r in json.loads((P/f'layers-device{d}.json').read_text())['records']]
assert sorted(r['layer'] for r in rows)==[0,3,24,27]
assert all(r['median_ms']['rounded']<r['median_ms']['w8a8'] for r in rows), 'NO_LAYER_GAIN'
print('REAL_LAYER_GATE_PASSED',flush=True)
