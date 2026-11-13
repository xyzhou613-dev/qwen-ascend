"""Archive an audited r152 outcome; install no live files."""
from pathlib import Path
import hashlib,json,shutil,subprocess,tarfile,urllib.request
P=Path(__file__).resolve().parent;R=Path('/root/qwen-ascend');LIVE=Path('/root/ascend-dev/fnx/fnx');B=P/'public-evidence'
o=json.load(open(P/'outcome.json'));deployed=o['status']=='deployed'
assert json.load(open(P/'final-operations.json'))['all_passed']
assert not any((P/n).exists() for n in ('lease.active','deployment.active','original-private.json'))
assert not subprocess.check_output(['git','status','--porcelain'],cwd=R).strip()
h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5));assert h['service_revision']==('r152' if deployed else 'r151') and h['worker_alive']
(P/'final-health.json').write_text(json.dumps(h,indent=2))
if deployed:
 assert json.load(open(P/'ab-summary.json'))['accepted'] and json.load(open(P/'production-verification.json'))['accepted']
 for x in json.load(open(P/'deployment-backup/manifest.json'))['entries']:assert hashlib.sha256(Path(x['target']).read_bytes()).hexdigest()==x['new_sha']
 for n in ('engine.py','server.py','state_update.py'):shutil.copyfile(LIVE/n,R/'fnx'/n)
 K=R/'kernels/state_update';K.mkdir(exist_ok=True);shutil.copyfile(P/'state_update.cpp',K/'state_update.cpp')
 (K/'build.sh').write_text((R/'kernels/l2norm_post/build.sh').read_text().replace('norm_post','state_update'));(K/'build.sh').chmod(0o755)
 shutil.copyfile(P/'r152-enable.py',R/'scripts/r152-enable.py')
from render_report import render
render(P,R,o)
B.mkdir(exist_ok=True)
names=['environment.json','report.md','outcome.json','ab-summary.json','final-operations.json','final-health.json','release-checks.json','baseline-reference-check.json','input-manifest.json','triple-long-manifest.json','quality-input-manifest.json','production-before-hashes.json','production-selected-config.json','candidate-source-hashes.json','baseline-quality.json','candidate-quality.json','baseline-health.json','candidate-health.json','baseline-startup.json','candidate-startup.json','component-device0.json','component-device1.json','long-graph-device0.json','long-graph-device1.json','delta-function-v2-device0.json','delta-function-v2-device1.json','delta-function-v3-device0.json','delta-function-v3-device1.json','all-linear-device0.json','all-linear-device1.json','layer-probe.json','baseline-config-check.json','candidate-config-check.json','fixtures.json','engine.patch','state_update.cpp','component_probe.py','component_launch.py','long_graph_probe.py','long_graph_launch.py','delta_function_probe.py','delta_function_launch.py','delta_function_inplace_probe.py','rejected-inplace-engine.py','rejected-inplace-wrapper.py','delta-function-initial-memory-limit.log','post-probe-limit-health.json','all_prefill_probe.py','all_prefill_launch.py','layer_probe.py','prefill_nll.py','http_bench.py','quality_client.py','evaluate.py','check_config.py','snapshot.py','stop_owned.py','launch.py','controller.sh','rescue.sh','deployment_files.py','deploy_stop.py','deploy.sh','deploy_rescue.sh','freeze_third_long.py','package_results.py','render_report.py','final_ops.py','r152-enable.py']
if deployed:names+=['production-health.json','production-verification.json']
else:names+=['post-restore-http.json']
for n in names:shutil.copyfile(P/n,B/n)
for folder in ('baseline','candidate','production','inputs'):
 if not (P/folder).exists():continue
 dest=B/folder;dest.mkdir(exist_ok=True)
 for f in (P/folder).glob('*.json'):shutil.copyfile(f,dest/f.name)
dest=B/'implementation';dest.mkdir(exist_ok=True)
for n in ('engine.py','state_update.py','libfnx_state_update.so'):shutil.copyfile(P/'candidate/fnx'/n,dest/n)
shutil.copyfile(P/'state_update.cpp',dest/'state_update.cpp');shutil.copyfile(P/'build.sh',dest/'build.sh')
(B/'README.md').write_text('''# r152 FP32 prefill state update

Read outcome.json, report.md, ab-summary.json and final-operations.json. A component speedup is not a whole-model speedup. Three distinct fresh 262016+128 requests per version, three fresh 8k requests and a cached repeat. Quality uses five frozen documents/995 real-prefill targets. The selected kernel is mode4: TILE4096, local decay cache, double-buffer input/output queues, separate FP32 multiply and add. No BF16 state or FMA. In-place state and other kernels are experimental evidence, not the selected production path. To reproduce inplace4, pair delta_function_inplace_probe.py with rejected-inplace-engine.py and rejected-inplace-wrapper.py in an isolated package; the final candidate has no in-place flag.

The flat short-graph timings did not resolve individual kernel costs; long_graph_probe.py captures256 updates with reused storage. The initial five-graph full-function probe hit its private1GiB allocator cap; the production service stayed healthy. The corrected probe retains only the required graphs. Operational helpers depend on fixed server paths and maintenance authorization. No full process ENV, credentials, weights, graph caches or real user activations are distributed.
''')
files=sorted(f for f in B.rglob('*') if f.is_file() and f.name!='SHA256SUMS')
(B/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(B))+'\n' for f in files))
with tarfile.open(P/'r152-state-prefill-evidence.tgz','w:gz') as t:t.add(B,arcname='r152-state-prefill-evidence')
shutil.copytree(B,R/'benchmarks/r152-state-prefill');print('R152_OUTCOME_PACKAGED',o['status'],len(files),flush=True)
