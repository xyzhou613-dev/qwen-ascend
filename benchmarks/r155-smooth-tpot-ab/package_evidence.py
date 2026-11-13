"""Explicit evidence allowlist; never recursively copy runtime or private ENV."""
import difflib,hashlib,json,shutil,tarfile
from pathlib import Path
P=Path(__file__).resolve().parent
repo=Path('/root/qwen-ascend');dest=repo/'benchmarks/r155-smooth-tpot-ab'
assert json.load(open(P/'final-operations.json'))['formal_http_smoke_passed']
assert json.load(open(P/'summary.json'))['production_restored']
assert not dest.exists(),'REFUSED_TO_OVERWRITE_EVIDENCE'
dest.mkdir()
scripts=['check_ready.py','controller.sh','evaluate.py','final_ops.py','formal_smoke.py','http_bench.py','idle_guard.py','launch.py','package_evidence.py','report.py','rescue.sh','safety.sh','selected_config.py','snapshot.py','stop_owned.py']
files=['summary.json','runtime-code-manifest.json','calibration-reference.json','input-manifest.json','quality-reference.json','production-before-hashes.json','final-operations.json','production-http.json','production-selected-config.json','ownership.json','idle-check.jsonl','controller.log','restore.log','resume-before-health.json','restored-health.json','candidate-smoke-repeatability.json','candidate-npu-state.txt','final-npu-state.txt']
for phase in ('baseline','candidate','rollback'):
 files += [f'{phase}-{suffix}' for suffix in ['health.json','loss16.json','server.log','startup.json','worker.log']]
 files += [f'{phase}/{name}' for name in ['8k.json','256k.json','cached256k.json','sustained1024.json','summary.json']]
files += ['candidate-projection-audit.json','baseline-reuse-check.json','http-timing-crosscheck.json','report.md']
files += [str(f.relative_to(P)) for f in P.glob('candidate-projections-*.jsonl')]
initial_names=['controller.sh','check_ready.py','safety.sh','controller.log','ownership.json','restore.log','restored-health.json','candidate-health.json','candidate-loss16.json','candidate-projection-audit.json','candidate-server.log','candidate-worker.log']
files += ['initial-attempt/'+n for n in initial_names]
files += [str(f.relative_to(P)) for f in (P/'initial-attempt').glob('candidate-projections-*.jsonl')]
for name in scripts+files:
 src=P/name;assert src.is_file(),name
 target=dest/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
for name,a,b in [('server-instrumentation.patch',Path('/root/ascend-dev/fnx/fnx/server.py'),P/'baseline/fnx/server.py'),('smooth-startup-audit.patch',Path('/models/fnx-diagnostics/r154-w8a8-gemv-20261113/selected/fnx/smooth_w8a8.py'),P/'candidate/fnx/smooth_w8a8.py')]:
 patch=''.join(difflib.unified_diff(a.read_text().splitlines(True),b.read_text().splitlines(True),fromfile='a/'+a.name,tofile='b/'+b.name))
 (dest/name).write_text(patch)
(dest/'README.md').write_text('''# r155 SmoothQuant W8A8 whole-model TPOT

See [report](../../docs/r155-smooth-tpot-ab.md). Research evidence only; production remains r152.

The initial A completed. Candidate startup was interrupted by the overly strict 1e-5 numerical sanity check; the original attempt is preserved. After canonical restoration, the resumed controller reused A and completed fresh B/A2. The candidate smoke tolerance is 0.001 nat; this does not change the r154 quality limits or establish final quality acceptance.

Runtime scripts use the original absolute diagnostic directory. `controller.sh` is the resumed B/A2 controller; the original A/B/A version is in `initial-attempt/`. These scripts temporarily stop the service and must not be launched casually. They require a saved original service snapshot and the independent rollback mechanisms described in the report. `launch.py` refuses a second Engine.

Candidate code is the r154 selected SmoothQuant implementation at f9284fa (`experiments/r154-w8a8-gemv`) plus the two instrumentation patches here. Runtime Python/ELF hashes are recorded in runtime-code-manifest.json. Input payloads remain in the r152 evidence directory and are identified by input-manifest.json; copy the exact six files from benchmarks/r152-state-prefill/inputs when reproducing. Calibration vectors stay in the server private directory; calibration-reference.json records their hashes and r154 provides the frozen calibration inputs and collection code.

`http_bench.py` records fresh input and output counts, SSE DONE, server TPOT and HTTP TTFT. The fresh medians use three different fixed inputs with 128 generated tokens each. Cache and sustained generation are separate single runs; 1024 is the requested limit, not guaranteed actual output. `evaluate.py` regenerates summary.json; `report.py` renders the report from these summaries and quality-reference.json.

No model weights, calibration vectors, hidden states, complete process ENV, credentials or graph caches are included.
''')
shutil.copyfile(P/'report.md',repo/'docs/r155-smooth-tpot-ab.md')
doc=repo/'docs/Qwen3.8-Flash-Next-2x910B4-技术文档.md'
old=doc.read_bytes();block=(P/'techdoc-block.md').read_bytes()
assert b'## r155 ' not in old[:1000],'ALREADY_UPDATED'
doc.write_bytes(block+old)
hashes={str(f.relative_to(dest)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(dest.rglob('*')) if f.is_file()}
(dest/'SHA256SUMS').write_text(''.join(f'{sha}  {name}\n' for name,sha in hashes.items()))
archive=P/'r155-smooth-tpot-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
 tf.add(dest,arcname='r155-smooth-tpot-evidence')
print('PACKAGED',len(hashes),'files',archive,hashlib.sha256(archive.read_bytes()).hexdigest())
