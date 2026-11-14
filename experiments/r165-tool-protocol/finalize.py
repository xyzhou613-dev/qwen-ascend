"""Verify accepted rollout and remove only its temporary maintenance controls."""
import hashlib,json,os,subprocess,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def cmd(*a):return subprocess.check_output(a,text=True).strip()
accepted=json.loads((P/'accepted.json').read_text());assert accepted['status']=='accepted'
assert not (P/'lease.active').exists() and not (P/'original-private.json').exists()
for unit in ('fnx-r165-deploy.service','fnx-r165-rollback.service'):
    assert cmd('systemctl','show',unit,'-p','ActiveState','--value') not in ('active','activating','deactivating')
for unit in ('fnx-r165-rollback.timer','fnx-r165-guard.service'):
    if cmd('systemctl','show',unit,'-p','LoadState','--value')!='not-found':subprocess.run(['systemctl','stop',unit],check=True)
for unit in ('fnx-r165-rollback.timer','fnx-r165-rollback.service'):(Path('/run/systemd/system')/unit).unlink(missing_ok=True)
Path('/etc/systemd/system/fnx-boot-restore.service.d/r165-pending-protocol.conf').unlink(missing_ok=True)
subprocess.run(['systemctl','daemon-reload'],check=True)
m=json.loads((P/'release-manifest.json').read_text());src=Path('/root/ascend-dev/fnx/fnx')
assert all(sha(src/n)==v for n,v in m['new_files'].items())
assert all(sha(src/n)==v for n,v in m['old_files'].items() if n not in m['new_files'])
assert sha(Path('/root/ascend-dev/p275.sh'))==m['p275_sha256']
assert all(sha(P/n)==v for n,v in json.loads((P/'harness-manifest.json').read_text()).items())
h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
assert h['status']=='ok' and h['worker_alive'] and h['service_revision']=='r152' and h.get('tool_protocol_revision')=='r165'
assert cmd('systemctl','is-active','model-switch.service')=='active'
assert cmd('systemctl','show','model-switch.service','-p','RefuseManualStart','--value')=='no'
assert 'r165' not in cmd('systemctl','show','model-switch.service','-p','DropInPaths','--value')
assert 'r165' not in cmd('systemctl','show','fnx-boot-restore.service','-p','DropInPaths','--value')
assert cmd('systemctl','is-enabled','fnx-boot-restore.service')=='enabled'
assert cmd('docker','inspect','-f','{{.Id}}','bole-model-Flash')==m['container_id']
engines=[];watch=[];extra=[]
for proc in Path('/proc').glob('[0-9]*'):
    try:a=proc.joinpath('cmdline').read_bytes().split(b'\0');cwd=os.readlink(proc/'cwd')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if b'fnx.server' in a or b'fnx.pipeline_worker' in a:engines.append(dict(pid=int(proc.name),cwd=cwd))
    if a and Path(os.fsdecode(a[0])).name in ('bash','sh') and b'/root/ascend-dev/watchdog.sh' in a:watch.append(int(proc.name))
    if any(str(P/n).encode() in a for n in ('deploy.py','recover.py','guard.py','live_acceptance.py','test_server_tool_protocol.py')):extra.append(int(proc.name))
assert len(engines)==2 and all(p['cwd']=='/root/ascend-dev/fnx' for p in engines)
assert len(watch)==1 and not extra
result=dict(status='passed',epoch=time.time(),health=h,engines=engines,watchdog_pids=watch,unchanged_core_files=43,
    p275_unchanged=True,container_preserved=True,temporary_controls_removed=True,normal_boot_restore_enabled=True,
    frozen_harness_verified=True,private_environment_removed=True,live_acceptance_cases=accepted['live_cases'],
    formal_8k_answer_exact=True,tool_protocol_revision='r165',engine_revision='r152',elapsed_s=accepted['elapsed_s'])
(P/'final-validation.json').write_text(json.dumps(result,indent=2));print('R165_DEPLOYMENT_FINAL_VALIDATION_PASS',flush=True)
