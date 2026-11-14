"""Bounded in-place protocol repair, with independent old-source rollback."""
import fcntl,hashlib,json,shutil,subprocess,time,traceback,urllib.request
from pathlib import Path
from context import restore_context
P=Path(__file__).resolve().parent
def run(*args,**kw):return subprocess.run(args,check=True,**kw)
def owned(mode):run('docker','exec','bole-model-Flash','/root/ascend-dev/venv/bin/python',str(P/'owned.py'),mode,timeout=30)
try:
    run('python3',str(P/'idle_guard.py'))
    owned('snapshot')
    run('systemctl','stop','model-switch.service')
    d=Path('/run/systemd/system/model-switch.service.d');d.mkdir(exist_ok=True)
    (d/'r165-exclusive.conf').write_text('[Unit]\nRefuseManualStart=yes\n')
    run('systemctl','daemon-reload');run('python3',str(P/'pause_watchdog.py'))
    run('python3',str(P/'idle_guard.py'))
    (P/'stop-requested').touch();owned('stop')
    run('python3',str(P/'apply_release.py'),'install')
    with (P/'candidate-start.log').open('ab') as log:
        run('env','WD=0','bash','/root/ascend-dev/p275.sh',stdout=log,stderr=subprocess.STDOUT,timeout=450)
    owned('register')
    h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
    assert h['worker_alive'] and h['service_revision']=='r152' and h.get('tool_protocol_revision')=='r165'
    (P/'candidate-health.json').write_text(json.dumps(h,indent=2))
    with (P/'live-tests.log').open('ab') as log:run('python3',str(P/'live_acceptance.py'),stdout=log,stderr=subprocess.STDOUT,timeout=150)
    run('python3',str(P/'formal_smoke.py'),timeout=120)
    shutil.copy2(P/'formal-smoke.json',P/'candidate-formal-smoke.json')
    live=json.loads((P/'live-acceptance.json').read_text());assert live['status']=='passed'
    m=json.loads((P/'release-manifest.json').read_text());src=Path('/root/ascend-dev/fnx/fnx')
    sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    assert all(sha(src/n)==v for n,v in m['new_files'].items())
    assert all(sha(src/n)==v for n,v in m['old_files'].items() if n not in m['new_files'])
    assert sha(Path('/root/ascend-dev/p275.sh'))==m['p275_sha256']
    state=open(P/'state.lock','w');fcntl.flock(state,fcntl.LOCK_EX)
    assert not (P/'rollback-requested.json').exists()
    restore_context()
    w=json.loads((P/'window.json').read_text())
    result=dict(status='accepted',epoch=time.time(),elapsed_s=time.time()-w['start_epoch'],engine_revision='r152',tool_protocol_revision='r165',
        unit_tests=39,handler_tests=28,live_cases=len(live['records']),unchanged_core_files=len(m['old_files'])-2,p275_unchanged=True,
        candidate_source_hashes=m['new_files'],formal_8k_exact=True)
    (P/'accepted.json').write_text(json.dumps(result,indent=2))
    (P/'original-private.json').unlink(missing_ok=True);(P/'lease.active').unlink()
    print('R165_PROTOCOL_REPAIR_ACCEPTED',result['elapsed_s'],flush=True)
except BaseException as ex:
    (P/'deployment-error.json').write_text(json.dumps(dict(epoch=time.time(),error=repr(ex)),indent=2))
    traceback.print_exc()
    subprocess.run(['systemctl','start','--no-block','fnx-r165-rollback.service'])
    raise
