"""Private verifier runtime with frozen SmoothQuant and prefix conv journals."""
import hashlib,json,shutil
from pathlib import Path
P=Path(__file__).resolve().parent
BASE=Path('/models/fnx-diagnostics/r154-w8a8-gemv-20261113/selected/fnx')
R153=Path('/models/fnx-diagnostics/r153-tpot-directions-20261113')
assert not (P/'runtime').exists()
dest=P/'runtime/fnx';shutil.copytree(BASE,dest,ignore=shutil.ignore_patterns('__pycache__'))
f=dest/'engine.py';s=f.read_text()
anchor='        seq = torch.cat([self.conv[0], mq.transpose(0, 1)], dim=-1)'
assert s.count(anchor)==1
s=s.replace(anchor,anchor+'''\n        if T <= 4 and os.environ.get("FNX_VERIFY_PREFIX_JOURNAL") == "1":
            from verifier_state import save_conv_prefixes
            save_conv_prefixes(self, seq, T)''',1)
f.write_text(s)
shutil.copyfile(R153/'small_delta.py',P/'small_delta.py')
for f in dest.glob('*.py'):compile(f.read_bytes(),str(f),'exec')
manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in dest.iterdir() if f.is_file() and f.suffix in ('.py','.so')}
(P/'runtime-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
live=Path('/root/ascend-dev/fnx/fnx')
(P/'production-before-hashes.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in live.iterdir() if f.is_file() and f.suffix in ('.py','.so')},indent=2)+'\n')
cal=Path('/models/fnx-diagnostics/r154-w8a8-gemv-20261113/private-calibration')
(P/'calibration-reference.json').write_text(json.dumps(dict(fixed_alpha=.5,recalibrated=False,files={f'L{i:02d}.safetensors':hashlib.sha256((cal/f'L{i:02d}.safetensors').read_bytes()).hexdigest() for i in (0,3,24,27)}),indent=2)+'\n')
print('R161_PRIVATE_RUNTIME_PREPARED_NO_ENGINE')
