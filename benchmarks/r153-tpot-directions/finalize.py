import hashlib,json,subprocess,tarfile,fnmatch,shutil,sys
from pathlib import Path
P=Path(__file__).resolve().parent
live=Path('/root/ascend-dev/fnx/fnx');repo=Path('/root/qwen-ascend')
def run(args):
    r=subprocess.run(args,text=True,capture_output=True,timeout=25)
    return dict(code=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr.strip())
before=json.loads((P/'production-before-hashes.json').read_text())
after={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in live.iterdir() if f.is_file() and f.suffix in ('.py','.so')}
import urllib.request
health=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=8))
helpers=[];watchdogs=[]
for f in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        cmd=f.read_bytes().replace(b'\0',b' ').decode(errors='replace')
        if '/models/fnx-diagnostics/r153-tpot-directions-20261113/' in cmd and ('probe.py' in cmd or 'launch.py' in cmd):
            helpers.append(dict(pid=int(f.parent.name),cmd=cmd))
        if cmd.strip() == '/bin/bash /root/ascend-dev/watchdog.sh':
            ppid=int(next(line.split()[1] for line in (f.parent/'status').read_text().splitlines() if line.startswith('PPid:')))
            watchdogs.append(dict(pid=int(f.parent.name),ppid=ppid,cmd=cmd))
    except (OSError,ValueError):pass
units=run(['systemctl','list-units','--all','--no-legend','--no-pager','fnx-r153-*'])
operations=dict(health=health,production_hashes_unchanged=before==after,production_file_count=len(after),
    hash_differences={k:dict(before=before.get(k),after=after.get(k)) for k in set(before)|set(after) if before.get(k)!=after.get(k)},
    helper_processes=helpers,watchdog_candidates=watchdogs,
    manager=run(['systemctl','show','model-switch.service','-p','ActiveState','-p','RefuseManualStart','-p','DropInPaths']),
    helper_units=units,npu=run(['npu-smi','info']),
    repo_head=run(['git','-C',str(repo),'rev-parse','HEAD']),
    current_date='2026-11-13',production_restarted=False,full_engine_loaded=False)
(P/'production-after-hashes.json').write_text(json.dumps(after,indent=2))
(P/'final-operations.json').write_text(json.dumps(operations,indent=2))
assert before==after and not helpers
assert health['service_revision']=='r152' and health['worker_alive'] and health['ctx']==262144
assert 'ActiveState=active' in operations['manager']['stdout']
assert 'RefuseManualStart=no' in operations['manager']['stdout']
assert not any(' active ' in line or ' activating ' in line for line in units['stdout'].splitlines())
assert len(watchdogs)==1 and watchdogs[0]['ppid']==1
if '--audit-only' in sys.argv:
    print('FINAL_AUDIT_PASS',len(after),watchdogs,operations['manager']['stdout'])
    raise SystemExit(0)

dest=repo/'benchmarks/r153-tpot-directions'
assert not dest.exists(),'new evidence directory already exists; inspect rather than overwrite'
dest.mkdir(parents=True)
patterns=['*.py','dense*.json','dense*.log','dense*.sh','verify*.json','verify*.log','verify*.sh',
    'hccl*.json','hccl*.log','quant-smoke*.log','small-delta-validation*.json',
    'mtp-inventory.json','production-selected-config.json','production-before-hashes.json',
    'production-after-hashes.json','post-layer-limit-health.json','npu-topology.txt',
    'summary.json','final-operations.json']
for f in P.iterdir():
    if f.is_file() and any(fnmatch.fnmatch(f.name,pat) for pat in patterns):shutil.copy2(f,dest/f.name)
(dest/'ge_compat').mkdir()
shutil.copy2(P/'ge_compat/sitecustomize.py',dest/'ge_compat/sitecustomize.py')
(dest/'README.md').write_text('''# r153 TPOT direction probes (not a deployed runtime)

See docs/r153-tpot-directions.md. Production stayed r152 throughout.

Scripts were executed inside bole-model-Flash with /root/ascend-dev/venv/bin/python.
Only launch.py selects production's whitelisted non-secret settings and one visible device;
hccl_verified_probe.py controls its own two ranks and unique TCP rendezvous port 29853.
All NPU timing runs were sequential, online and non-exclusive. Do not overlap them.
The scripts deliberately guard r152 and free HBM; inspect current ownership, versions,
memory headroom and process state before any future reuse. Never load another Engine.

CPU analysis: python analyze.py (expects the adjacent JSON files).
To reproduce NZ format creation, use an isolated private dependency directory; successful
probes used decorator, scipy, attrs and psutil plus ge_compat/sitecustomize.py for compiler
subprocess NumPy aliases. A SciPy/NumPy version warning remained. This is diagnostic
compatibility, not an approved production dependency change. Dependencies and pip logs
are deliberately excluded from the evidence package. No global pip operation was run.

dense_probe.py: hot single-weight ND layout comparison.
dense_stream_probe.py: >=384 MiB rotating-weight streaming proxy (ND).
dense_stream_nz_probe.py: same streaming proxy with verified format-29 replicas.
hccl_probe.py: initial timing; hccl_verified_probe.py adds fresh nonzero graph-sum verification.
verify_layer_probe.py: initial generic-512-group allocator-cap failure, retained as negative evidence.
verify_layer_small_probe.py: active/padded expert groups, graph-safe and bounded.
verify_layer_loop_probe.py: tokenwise MoE with batched attention/HC.
verify_layer_native_probe.py + small_delta.py: T<=4 native FP32 recurrence with prefix-state journal.

Synthetic inputs, four sampled real layers and component timing do not constitute model
quality or end-to-end TPOT acceptance. No model weights, user activations, full process
environment snapshots, graph caches, private dependency binaries or credentials included.
''')
shutil.copy2(P/'report.md',repo/'docs/r153-tpot-directions.md')
doc=repo/'docs/Qwen3.8-Flash-Next-2x910B4-技术文档.md'
old=doc.read_text(encoding='utf-8-sig')
assert '## r153 方向探索更新' not in old
doc.write_text((P/'techdoc-block.md').read_text()+old,encoding='utf-8')
files=sorted(f for f in dest.rglob('*') if f.is_file())
(dest/'SHA256SUMS').write_text(''.join(f'{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(dest).as_posix()}\n' for f in files))
archive=P/'r153-tpot-directions-evidence.tgz'
with tarfile.open(archive,'w:gz') as t:
    t.add(dest,arcname='r153-tpot-directions-evidence')
    t.add(P/'report.md',arcname='r153-tpot-directions-evidence/report.md')
(P/'r153-tpot-directions-evidence.tgz.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest()+'  '+archive.name+'\n')
print(json.dumps(dict(files=len(files),archive_bytes=archive.stat().st_size,hashes_unchanged=True,revision=health['service_revision'],units=units['stdout'],manager=operations['manager']['stdout']),indent=2))
