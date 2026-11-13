"""Package only the isolated numeric experiment, after the main run is restored."""
import hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent
repo=Path('/root/qwen-ascend');exp=repo/'experiments/r158-bf16-down';bench=repo/'benchmarks/r158-bf16-down'
assert json.load(open('/models/fnx-diagnostics/r157-combined-quality-20261113/final-validation.json'))['private_env_removed']
assert not exp.exists() and not bench.exists()
for name in ('prepare_variant.py','build.sh','numeric_probe.py','bf16_weight_down.cpp','libfnx_w4_bf16w.so','variant.patch','README.md','package.py'):
 exp.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/name,exp/name)
for name in ('source-manifest.json','numeric-result.json','build.log','numeric.log','initial-visibility-error.log'):
 bench.mkdir(parents=True,exist_ok=True);shutil.copyfile(P/name,bench/name)
for root in (exp,bench):
 files=sorted(f for f in root.iterdir() if f.is_file())
 (root/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+f.name+'\n' for f in files))
print('NUMERIC_EXPERIMENT_PACKAGED')
