"""Read/compile only; no accelerator initialization and no service mutation."""
import hashlib,json,subprocess,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
assert not (P/'window-start.txt').exists()
for f in P.rglob('*.py'):compile(f.read_bytes(),str(f),'exec')
for f in P.glob('*.sh'):subprocess.run(['bash','-n',str(f)],check=True)
runtime=json.loads((P/'runtime-code-manifest.json').read_text())
for phase,files in runtime.items():
    assert all(hashlib.sha256((P/phase/'fnx'/name).read_bytes()).hexdigest()==sha for name,sha in files.items())
before=json.loads((P/'production-before-hashes.json').read_text())
live=Path('/root/ascend-dev/fnx/fnx')
assert before=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in live.iterdir() if f.is_file() and f.suffix in ('.py','.so')}
assert len(before)==45 and (P/'p275-before.sh').read_bytes()==Path('/root/ascend-dev/p275.sh').read_bytes()
assert subprocess.check_output(['git','status','--porcelain'],cwd='/root/qwen-ascend')==b''
h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
assert h['status']=='ok' and h['service_revision']=='r152' and h['worker_alive']
assert subprocess.check_output(['systemctl','is-active','model-switch.service'],text=True).strip()=='active'
manifest={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in P.iterdir() if f.suffix in ('.py','.sh')}
for name in ('debug-policy.json','runtime-code-manifest.json'):manifest[name]=hashlib.sha256((P/name).read_bytes()).hexdigest()
(P/'harness-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
(P/'preflight.json').write_text(json.dumps(dict(production_files_unchanged=45,python_compile_pass=True,
    bash_syntax_pass=True,formal_health=h,manager_active=True,model_started=False,
    service_interrupted=False,source_files_frozen=len(manifest)),indent=2)+'\n')
print('CPU_PREFLIGHT_PASSED_NO_MODEL_LOADED_NO_INTERRUPTION',len(manifest))
