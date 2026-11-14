"""After independent restoration; verify and clean this experiment only."""
import hashlib,json,os,subprocess,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
def cmd(*args):return subprocess.check_output(args,text=True).strip()
assert not (P/'lease.active').exists() and not (P/'original-private.json').exists()
for unit in ('fnx-r162-run.service','fnx-r162-rescue.service'):
    assert cmd('systemctl','show',unit,'-p','ActiveState','--value') not in ('active','activating','deactivating')
for unit in ('fnx-r162-rescue.timer','fnx-r162-safety.service'):
    if cmd('systemctl','show',unit,'-p','LoadState','--value')!='not-found':subprocess.run(['systemctl','stop',unit],check=True)
for unit in ('fnx-r162-rescue.timer','fnx-r162-rescue.service'):(Path('/run/systemd/system')/unit).unlink(missing_ok=True)
subprocess.run(['systemctl','daemon-reload'],check=True)
h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
assert h['status']=='ok' and h['service_revision']=='r152' and h['worker_alive'] and 'diagnostic_variant' not in h
assert cmd('systemctl','is-active','model-switch.service')=='active'
assert cmd('systemctl','show','model-switch.service','-p','RefuseManualStart','--value')=='no'
assert 'r162' not in cmd('systemctl','show','model-switch.service','-p','DropInPaths','--value')
before=json.loads((P/'production-before-hashes.json').read_text());live=Path('/root/ascend-dev/fnx/fnx')
assert before=={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in live.iterdir() if f.is_file() and f.suffix in ('.py','.so')}
assert (P/'p275-before.sh').read_bytes()==Path('/root/ascend-dev/p275.sh').read_bytes()
for phase,files in json.loads((P/'runtime-code-manifest.json').read_text()).items():
    assert all(hashlib.sha256((P/phase/'fnx'/name).read_bytes()).hexdigest()==sha for name,sha in files.items())
frozen=json.loads((P/'harness-manifest.json').read_text())
assert all(hashlib.sha256((P/name).read_bytes()).hexdigest()==sha for name,sha in frozen.items())
engines=[];watchdogs=[];experiment=[]
for p in Path('/proc').glob('[0-9]*'):
    try:argv=p.joinpath('cmdline').read_bytes().split(b'\0');cwd=os.readlink(p/'cwd')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if b'fnx.server' in argv or b'fnx.pipeline_worker' in argv:engines.append(dict(pid=int(p.name),cwd=cwd))
    if argv and Path(os.fsdecode(argv[0])).name in ('bash','sh') and b'/root/ascend-dev/watchdog.sh' in argv:watchdogs.append(int(p.name))
    if cwd in (str(P/'baseline'),str(P/'smooth'),str(P/'combined')):experiment.append(int(p.name))
    if any(str(P/name).encode() in argv for name in ('controller.sh','safety.py','quality_eval.py','quality_client.py')):experiment.append(int(p.name))
assert len(engines)==2 and all(x['cwd']=='/root/ascend-dev/fnx' for x in engines)
assert len(watchdogs)==1 and not experiment
run_id=(P/'run-id').read_text().strip()
assert not cmd('docker','ps','-q','--filter','label=fnx.diagnostic=r157','--filter','label=fnx.run='+run_id)
assert cmd('systemctl','is-enabled','fnx-boot-restore.service')=='enabled'
assert cmd('docker','inspect','-f','{{.Id}}','bole-model-Flash')==json.loads(Path('/root/ascend-dev/fnx-boot-restore/container-identity.json').read_text())['container_id']
subprocess.run(['/usr/bin/python3',str(P/'formal_smoke.py')],check=True)
restore=json.loads((P/'restore-complete.json').read_text());assert restore['within_six_hours']
result=dict(epoch=time.time(),production_files_unchanged=45,p275_unchanged=True,health=h,engines=engines,
    watchdog_pids=watchdogs,manager_active=True,temporary_controls_removed=True,private_env_removed=True,
    experimental_processes_absent=True,runtime_and_harness_unchanged=True,original_container_preserved=True,
    formal_fresh8k_answer_identical=True,within_six_hour_cap=True,restore_elapsed_s=restore['elapsed_s'])
(P/'final-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('R162_FORMAL_RESTORE_AND_INPUT_LIFECYCLE_VERIFIED',flush=True)
