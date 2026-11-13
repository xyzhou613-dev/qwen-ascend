"""Verify canonical restoration after this component-only continuation."""
import hashlib,json,os,subprocess,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
def cmd(*args):return subprocess.check_output(args,text=True).strip()
assert not (P/'lease.active').exists() and not (P/'original-private.json').exists()
for unit in ('fnx-r160-run.service','fnx-r160-rescue.service'):
    assert cmd('systemctl','show',unit,'-p','ActiveState','--value') not in ('active','activating','deactivating')
for unit in ('fnx-r160-rescue.timer','fnx-r160-safety.service'):
    if cmd('systemctl','show',unit,'-p','LoadState','--value')!='not-found':
        subprocess.run(['systemctl','stop',unit],check=True)
for unit in ('fnx-r160-rescue.timer','fnx-r160-rescue.service'):
    (Path('/run/systemd/system')/unit).unlink(missing_ok=True)
subprocess.run(['systemctl','daemon-reload'],check=True)
h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
assert h['status']=='ok' and h['service_revision']=='r152' and h['worker_alive'] and 'diagnostic_variant' not in h
assert cmd('systemctl','is-active','model-switch.service')=='active'
assert cmd('systemctl','show','model-switch.service','-p','RefuseManualStart','--value')=='no'
assert 'r160' not in cmd('systemctl','show','model-switch.service','-p','DropInPaths','--value')
assert not Path('/run/systemd/system/model-switch.service.d/r160-exclusive.conf').exists()
expected=json.loads((P/'production-before-hashes.json').read_text());live=Path('/root/ascend-dev/fnx/fnx')
actual={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in live.iterdir() if f.is_file() and f.suffix in ('.py','.so')}
assert actual==expected and len(actual)==45
assert (P/'p275-before.sh').read_bytes()==Path('/root/ascend-dev/p275.sh').read_bytes()
frozen=json.loads((P/'source-manifest.json').read_text())
assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in frozen.items())
watchdogs=[];engines=[];extra=[]
for p in Path('/proc').glob('[0-9]*'):
    try:argv=p.joinpath('cmdline').read_bytes().split(b'\0');cwd=os.readlink(p/'cwd')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if b'fnx.server' in argv or b'fnx.pipeline_worker' in argv:engines.append(dict(pid=int(p.name),cwd=cwd))
    if argv and Path(os.fsdecode(argv[0])).name in ('bash','sh') and b'/root/ascend-dev/watchdog.sh' in argv:watchdogs.append(int(p.name))
    if any(str(P/name).encode() in argv for name in ('probe.py','launch.py','micro_controller.sh','safety.py')):extra.append(int(p.name))
assert len(engines)==2 and all(e['cwd']=='/root/ascend-dev/fnx' for e in engines)
assert len(watchdogs)==1 and not extra,(watchdogs,extra)
assert cmd('systemctl','is-enabled','fnx-boot-restore.service')=='enabled'
cid=cmd('docker','inspect','-f','{{.Id}}','bole-model-Flash')
assert cid==json.loads(Path('/root/ascend-dev/fnx-boot-restore/container-identity.json').read_text())['container_id']
for d in (0,1):
    for mode in ('numeric','timing'):
        path=P/f'{mode}-device{d}.json'
        if path.exists():
            row=json.loads(path.read_text())
            assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in row['code_sha256'].items())
subprocess.run(['/usr/bin/python3',str(P/'formal_smoke.py')],check=True)
smoke=json.loads((P/'production-http.json').read_text());assert smoke['answer_matches_baseline']
restore=json.loads((P/'restore-complete.json').read_text());window=json.loads((P/'window.json').read_text())
assert restore['epoch']<=window['hard_cap_epoch'] and restore['within_hour']
result=dict(epoch=time.time(),production_revision='r152',health=h,production_files_unchanged=45,p275_unchanged=True,
    original_container_preserved=True,source_and_binary_frozen=True,manager_active=True,watchdog_pids=watchdogs,
    engines=engines,private_env_removed=True,temporary_controls_removed=True,experimental_processes_absent=True,
    normal_boot_recovery_enabled=True,formal_fresh8k_answer_identical=True,
    restore_elapsed_s_from_original_window=restore['elapsed_s'],
    micro_elapsed_s=restore['epoch']-window['micro_started_epoch'],original_hard_cap_preserved=True)
(P/'final-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('R160_FINAL_RESTORE_VERIFIED',result['micro_elapsed_s'],'seconds; 45 files unchanged')
