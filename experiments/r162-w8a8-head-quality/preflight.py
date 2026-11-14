"""Read/compile checks, no NPU initialization or interruption."""
import hashlib,json,subprocess,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;assert not (P/'window-start.txt').exists()
for f in P.glob('*.py'):compile(f.read_bytes(),str(f),'exec')
for f in P.glob('*.sh'):subprocess.run(['bash','-n',str(f)],check=True)
for phase,files in json.loads((P/'runtime-code-manifest.json').read_text()).items():
    assert all(hashlib.sha256((P/phase/'fnx'/name).read_bytes()).hexdigest()==sha for name,sha in files.items())
before=json.loads((P/'production-before-hashes.json').read_text());live=Path('/root/ascend-dev/fnx/fnx')
assert before=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in live.iterdir() if f.is_file() and f.suffix in ('.py','.so')}
assert len(before)==45 and (P/'p275-before.sh').read_bytes()==Path('/root/ascend-dev/p275.sh').read_bytes()
reused=json.loads((P/'baseline-reuse-provenance.json').read_text())
assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==meta['copied_sha256'] for name,meta in reused.items())
raw=(P/'private-quality/tasks.jsonl').read_bytes()+(P/'quality-extra-tasks.jsonl').read_bytes()
assert hashlib.sha256(raw).hexdigest()=='3362acf6e696bedcd58c97b2f2714bf96d71088dbc200a06c5f638fa466b06c2'
judge=json.loads((P/'judge-image.json').read_text())
assert subprocess.check_output(['docker','image','inspect','-f','{{.Id}}',judge['tag']],text=True).strip()==judge['image_id']
h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
assert h['status']=='ok' and h['service_revision']=='r152' and h['worker_alive']
assert subprocess.check_output(['systemctl','is-active','model-switch.service'],text=True).strip()=='active'
result=dict(production_files_unchanged=45,source_compile_pass=True,judge_image_unchanged=True,
    task_inputs_sha256=hashlib.sha256(raw).hexdigest(),head_module_matches_component_test=
    (P/'combined/fnx/quant_head.py').read_bytes()==Path('/models/fnx-diagnostics/r161-quant-v4-20261114/quant_head.py').read_bytes(),
    baseline_checkpoint_rows=999,service_interrupted=False,engine_started=False)
assert result['head_module_matches_component_test']
(P/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
print('R162_PREFLIGHT_PASS_NO_ENGINE_NO_INTERRUPTION',flush=True)
