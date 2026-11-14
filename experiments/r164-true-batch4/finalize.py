"""Read-only production checks, remove only r164 transient controls, smoke."""
import hashlib,json,os,subprocess,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
def cmd(*a):return subprocess.check_output(a,text=True).strip()
assert (P/'restore-complete.json').exists()
restore=json.loads((P/'restore-complete.json').read_text());assert restore['within_hour']
assert not (P/'lease.active').exists() and not (P/'original-private.json').exists()
for unit in ('fnx-r164-run.service','fnx-r164-run-nd.service','fnx-r164-rescue.service'):
    assert cmd('systemctl','show',unit,'-p','ActiveState','--value') not in ('active','activating','deactivating')
for unit in ('fnx-r164-rescue.timer','fnx-r164-safety.service'):
    if cmd('systemctl','show',unit,'-p','LoadState','--value')!='not-found':subprocess.run(['systemctl','stop',unit],check=True)
for unit in ('fnx-r164-rescue.timer','fnx-r164-rescue.service'):(Path('/run/systemd/system')/unit).unlink(missing_ok=True)
subprocess.run(['systemctl','daemon-reload'],check=True)
h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
assert h['status']=='ok' and h['service_revision']=='r152' and h['worker_alive'] and not h.get('diagnostic_variant')
assert cmd('systemctl','is-active','model-switch.service')=='active'
assert cmd('systemctl','show','model-switch.service','-p','RefuseManualStart','--value')=='no'
assert 'r164' not in cmd('systemctl','show','model-switch.service','-p','DropInPaths','--value')
assert cmd('systemctl','is-enabled','fnx-boot-restore.service')=='enabled'
expected=json.loads((P/'canonical-manifest.json').read_text());base=Path('/root/ascend-dev/fnx/fnx')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
assert all(sha(base/k)==v for k,v in expected['files'].items())
assert sha(Path('/root/ascend-dev/p275.sh'))==expected['p275']
frozen=json.loads((P/'harness-manifest.json').read_text())
assert all(sha(P/k)==v for k,v in frozen.items()),'FROZEN_HARNESS_CHANGED'
nd=json.loads((P/'nd-manifest.json').read_text())
assert all(sha(P/k)==v for k,v in nd.items()),'SUPPLEMENTAL_HARNESS_CHANGED'
engines=[];watchdogs=[];experiment=[]
for p in Path('/proc').glob('[0-9]*'):
    try:argv=p.joinpath('cmdline').read_bytes().split(b'\0');cwd=os.readlink(p/'cwd')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if b'fnx.server' in argv or b'fnx.pipeline_worker' in argv:engines.append(dict(pid=int(p.name),cwd=cwd))
    if argv and Path(os.fsdecode(argv[0])).name in ('bash','sh') and b'/root/ascend-dev/watchdog.sh' in argv:watchdogs.append(int(p.name))
    if any(str(P/n).encode() in argv for n in ('bench.py','bench_nd.py','launch_bench.py','launch_nd.py','probe.py','probe_nd.py','launch_probe.py','controller.sh','controller_nd.sh','safety.py')):experiment.append(int(p.name))
assert len(engines)==2 and all(e['cwd']=='/root/ascend-dev/fnx' for e in engines)
assert len(watchdogs)==1 and not experiment
container=cmd('docker','inspect','-f','{{.Id}}','bole-model-Flash')
assert container==json.loads(Path('/root/ascend-dev/fnx-boot-restore/container-identity.json').read_text())['container_id']
for unit in ('fnx-r164-run.service','fnx-r164-run-nd.service','fnx-r164-probe-nd.service'):
    subprocess.run(['systemctl','reset-failed',unit],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
subprocess.run(['python3',str(P/'formal_smoke.py')],check=True)
result=dict(status='passed',epoch=time.time(),restore_elapsed_s=restore['elapsed_s'],within_hour=True,
    production_files_unchanged=len(expected['files']),p275_unchanged=True,
    original_container_preserved=True,health=h,engines=engines,watchdog_pids=watchdogs,
    manager_active=True,normal_boot_restore_enabled=True,temporary_controls_removed=True,
    no_experimental_processes=True,private_environment_removed=True,frozen_harness_verified=True,
    formal_fresh8k_answer_exact=True)
(P/'final-validation.json').write_text(json.dumps(result,indent=2))
print('R164_FORMAL_RESTORE_VERIFIED',restore['elapsed_s'])
