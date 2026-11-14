"""Read-only verification after bounded online component probes."""
import hashlib,json,os,subprocess,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cmd(*a):return subprocess.check_output(a,text=True).strip()
live=Path('/root/ascend-dev/fnx/fnx')
before=json.loads((P/'production-before-hashes.json').read_text())
assert len(before)==45 and before=={f.name:sha(f) for f in live.iterdir() if f.is_file() and f.suffix in ('.py','.so')}
assert Path('/root/ascend-dev/p275.sh').read_bytes()==Path('/models/fnx-diagnostics/r160-ub-lut-down-20261114/p275-before.sh').read_bytes()
for name in ('harness-manifest.json','ablation-harness-manifest.json','head-harness-manifest.json','rollback-harness-manifest.json'):
    assert all(sha(P/f)==h for f,h in json.loads((P/name).read_text()).items()),name
assert all(sha(P/'runtime/fnx'/f)==h for f,h in json.loads((P/'runtime-manifest.json').read_text()).items())
cal=json.loads((P/'calibration-reference.json').read_text())
assert cal['fixed_alpha']==.5 and not cal['recalibrated']
calroot=Path('/models/fnx-diagnostics/r154-w8a8-gemv-20261113/calibration')
if not calroot.exists():
    candidates=list(Path('/models/fnx-diagnostics/r154-w8a8-gemv-20261113').rglob('L00.safetensors'))
    assert len(candidates)==1,candidates
    calroot=candidates[0].parent
assert all(sha(calroot/f)==h for f,h in cal['files'].items())
engines=[];watchdogs=[];probes=[]
for p in Path('/proc').glob('[0-9]*'):
    try:argv=p.joinpath('cmdline').read_bytes().split(b'\0');cwd=os.readlink(p/'cwd')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    if b'fnx.server' in argv or b'fnx.pipeline_worker' in argv:engines.append(dict(pid=int(p.name),cwd=cwd))
    if argv and Path(os.fsdecode(argv[0])).name in ('bash','sh') and b'/root/ascend-dev/watchdog.sh' in argv:watchdogs.append(int(p.name))
    if any(str(P/n).encode() in argv for n in ('probe.py','ablation.py','head_probe.py','rollback_probe.py')):probes.append(int(p.name))
assert {e['pid'] for e in engines}=={141969,141988} and all(e['cwd']=='/root/ascend-dev/fnx' for e in engines)
assert len(watchdogs)==1 and not probes
for label in ('device','ablation','head','rollback'):
    for d in (0,1):assert cmd('systemctl','show',f'fnx-r161-{label}{d}.service','-p','ActiveState','--value') not in ('active','activating','deactivating')
h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
assert h['status']=='ok' and h['service_revision']=='r152' and h['worker_alive'] and 'diagnostic_variant' not in h
assert h['pipeline_transfers']==dict(peer_transfers=6,peer_bytes=162529280,host_transfers=132,host_bytes=5222400)
assert cmd('systemctl','is-active','model-switch.service')=='active'
assert cmd('systemctl','show','model-switch.service','-p','RefuseManualStart','--value')=='no'
assert 'r161' not in cmd('systemctl','show','model-switch.service','-p','DropInPaths','--value')
assert cmd('systemctl','is-enabled','fnx-boot-restore.service')=='enabled'
assert cmd('docker','inspect','-f','{{.Id}}','bole-model-Flash')=='52460e4d0cd22dc5550f6403e093afa057c589aeab6168563a55daf108f53862'
result=dict(epoch=time.time(),production_files_unchanged=45,p275_unchanged=True,health=h,
    engines=engines,original_pids_preserved=True,service_interrupted=False,full_engine_started=False,
    watchdog_pids=watchdogs,manager_active=True,production_transfers_stable=True,
    runtime_harness_calibration_unchanged=True,probe_processes_absent=True,original_container_preserved=True,
    boot_restore_enabled=True,experimental_lease_or_service_controls_created=False)
(P/'final-validation.json').write_text(json.dumps(result,indent=2)+'\n')
print('R161_ONLINE_COMPONENT_LIFECYCLE_PASS_NO_RESTART')
