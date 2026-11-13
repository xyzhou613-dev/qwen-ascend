"""Predeclared component gate; never choose a tile from quality outcomes."""
import json,math
from pathlib import Path
P=Path(__file__).resolve().parent
rows=[json.loads((P/f'component-device{d}.json').read_text()) for d in (0,1)]
assert all(r['status']=='passed_component' and r['experts_covered']==512 for r in rows)
ratios={str(tile):[r['median_us'][f'rounded{tile}']/r['median_us']['production'] for r in rows] for tile in (128,256)}
valid={tile:r for tile,r in ratios.items() if max(r)<=.98}
result=dict(ratios=ratios,passed=bool(valid),selected_tile=min(valid,key=lambda k:math.prod(valid[k])) if valid else None)
(P/'component-selection.json').write_text(json.dumps(result,indent=2)+'\n')
assert valid, 'NO_COMPONENT_GAIN: stop before whole-model testing'
(P/'selected-tile.txt').write_text(result['selected_tile']+'\n')
print('COMPONENT_GATE',result,flush=True)
