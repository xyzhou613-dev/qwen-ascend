"""Only relax a guard in the private snapshot; all production bytes retained."""
from pathlib import Path
P=Path(__file__).resolve().parent
p=P/'runtime/fnx/engine.py'
s=p.read_text()
old='if B != 1:\n                raise ValueError("QSA indexer currently requires FNX_BATCH=1")'
new='if B != 1 and os.environ.get("FNX_EXPERIMENT_BATCH_ROWS") != "1":\n                raise ValueError("QSA indexer currently requires FNX_BATCH=1")'
assert s.count(old)==1
p.write_text(s.replace(old,new))
