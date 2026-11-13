"""Run after independent rescue finishes; verify and clear only owned controls."""
import hashlib,json,os,subprocess,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
def cmd(*args):return subprocess.check_output(args,text=True).strip()
h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
assert h['status']=='ok' and h['service_revision']=='r152' and h['worker_alive'] and 'diagnostic_variant' not in h
assert not (P/'lease.active').exists() and not (P/'original-private.json').exists()
assert cmd('systemctl','show','fnx-r159-run.service','-p','ActiveState','--value') not in ('active','activating','deactivating')
assert cmd('systemctl','show','fnx-r159-rescue.service','-p','ActiveState','--value') not in ('active','activating','deactivating')
for unit in ('fnx-r159-rescue.timer','fnx-r159-safety.service'):
    if cmd('systemctl','show',unit,'-p','LoadState','--value')!='not-found':
        subprocess.run(['systemctl','stop',unit],check=True)
for name in ('fnx-r159-rescue.timer','fnx-r159-rescue.service'):
    path=Path('/run/systemd/system')/name
    if path.exists():path.unlink()
subprocess.run(['systemctl','daemon-reload'],check=True)
manifest=json.loads((P/'harness-manifest.json').read_text())
assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items()),'harness changed'
runtime=json.loads((P/'runtime-code-manifest.json').read_text())
for phase,files in runtime.items():
    assert all(hashlib.sha256((P/phase/'fnx'/name).read_bytes()).hexdigest()==sha for name,sha in files.items())
before=json.loads((P/'production-before-hashes.json').read_text());live=Path('/root/ascend-dev/fnx/fnx')
after={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in live.iterdir() if f.is_file() and f.suffix in ('.py','.so')}
assert before==after and len(after)==45
assert (P/'p275-before.sh').read_bytes()==Path('/root/ascend-dev/p275.sh').read_bytes()
assert cmd('systemctl','is-active','model-switch.service')=='active'
assert cmd('systemctl','show','model-switch.service','-p','RefuseManualStart','--value')=='no'
assert 'r159' not in cmd('systemctl','show','model-switch.service','-p','DropInPaths','--value')
assert not Path('/run/systemd/system/model-switch.service.d/r159-exclusive.conf').exists()
watchdogs=[];engines=[];experimental=[]
for p in Path('/proc').glob('[0-9]*'):
    try:
        argv=p.joinpath('cmdline').read_bytes().split(b'\0');cwd=os.readlink(p/'cwd')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if b'fnx.server' in argv or b'fnx.pipeline_worker' in argv:engines.append(dict(pid=int(p.name),cwd=cwd))
    if argv and Path(os.fsdecode(argv[0])).name in ('bash','sh') and b'/root/ascend-dev/watchdog.sh' in argv:watchdogs.append(int(p.name))
    if cwd in (str(P/'baseline'),str(P/'combined')) and (b'fnx.server' in argv or b'fnx.pipeline_worker' in argv):experimental.append(int(p.name))
    if any(str(P/name).encode() in argv for name in ('component_probe.py','layer_probe.py','controller.sh','safety.py')):experimental.append(int(p.name))
assert len(engines)==2 and all(e['cwd']=='/root/ascend-dev/fnx' for e in engines)
assert len(watchdogs)==1 and not experimental
assert cmd('systemctl','is-enabled','fnx-boot-restore.service')=='enabled'
restore=json.loads((P/'restore-complete.json').read_text())
assert restore['within_hour']
subprocess.run(['/usr/bin/python3',str(P/'formal_smoke.py')],check=True)
smoke=json.loads((P/'production-http.json').read_text())
assert smoke['answer_matches_baseline'] and smoke['stats']['tokens']==128 and smoke['stats']['n_prefill']>=8060
result=dict(verified_epoch=time.time(),health=h,production_files_unchanged=len(after),p275_unchanged=True,
    runtime_and_harness_unchanged=True,original_container_preserved=cmd('docker','inspect','-f','{{.Id}}','bole-model-Flash')==json.loads(Path('/root/ascend-dev/fnx-boot-restore/container-identity.json').read_text())['container_id'],
    formal_fresh_8k_smoke_passed=True,manager_active=True,watchdog_pids=watchdogs,engines=engines,
    temporary_controls_removed=True,experimental_processes_absent=True,private_env_removed=True,
    restore_elapsed_s=restore['elapsed_s'],within_hour=True)
assert result['original_container_preserved']
(P/'final-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('FINAL_RESTORE_VERIFIED',len(after),'unchanged files; fresh8k answer identical')
