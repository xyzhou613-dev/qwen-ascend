"""Read-only audit after the normal watchdog has been handed back."""
from pathlib import Path
import hashlib, json, socket, subprocess, urllib.request
P = Path('/models/fnx-diagnostics/r152-state-prefill-20261113')
def out(*cmd): return subprocess.check_output(cmd, text=True).strip()
deployed=json.load(open(P/'outcome.json'))['status']=='deployed'
checks = {}
checks['manager_active'] = out('systemctl','is-active','model-switch.service') == 'active'
checks['manager_start_allowed'] = out('systemctl','show','model-switch.service','-p','RefuseManualStart','--value') == 'no'
units = ['fnx-r152-controller.service','fnx-r152-rescue.timer','fnx-r152-deploy.service',
         'fnx-r152-deploy-rescue.timer','fnx-r151-watchdog.service','fnx-r152-watchdog.service','fnx-r152-component.service','fnx-r152-layout.service','fnx-r152-all-prefill.service','fnx-r152-long-graph.service','fnx-r152-delta-function.service','fnx-r152-delta-function-v2.service','fnx-r152-delta-function-v3.service']
states = {u:out('systemctl','show',u,'-p','ActiveState','--value') for u in units}
checks['temporary_units_inactive'] = all(v in ('inactive','failed','') for v in states.values())
checks['private_files_removed'] = not any((P/n).exists() for n in ['lease.active','deployment.active','original-private.json'])
checks['dropin_removed'] = not Path('/run/systemd/system/model-switch.service.d/r152-exclusive.conf').exists()
watchdogs = []
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        cmd = proc.joinpath('cmdline').read_bytes().split(b'\0')
        if b'/root/ascend-dev/watchdog.sh' in cmd:
            parent = int(next(s.split()[1] for s in proc.joinpath('status').read_text().splitlines() if s.startswith('PPid:')))
            watchdogs.append(dict(pid=int(proc.name),ppid=parent))
    except (FileNotFoundError,ProcessLookupError,PermissionError): pass
checks['one_normal_watchdog'] = len(watchdogs)==1 and watchdogs[0]['ppid']==1
sock=socket.socket();sock.settimeout(1)
checks['diagnostic_port_closed'] = sock.connect_ex(('127.0.0.1',19002)) != 0
sock.close()
health=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
checks['production_ready'] = health['service_revision']==('r152' if deployed else 'r151') and health['worker_alive'] and health.get('state_update',False)==deployed and health['norm_ordered'] and health['prefill_output'] and health['ctx']==262144
live=Path('/root/ascend-dev/fnx/fnx')
if deployed:
 m=json.load(open(P/'deployment-backup/manifest.json'))
 checks['release_hashes_match']=all(hashlib.sha256(Path(x['target']).read_bytes()).hexdigest()==x['new_sha'] for x in m['entries'])
 changed={'engine.py','server.py'}
else:
 checks['startup_unchanged']=Path('/root/ascend-dev/p275.sh').read_bytes()==(P/'p275-before.sh').read_bytes()
 checks['candidate_not_installed']=not (live/'state_update.py').exists() and not (live/'libfnx_state_update.so').exists()
 changed=set()
checks['other_production_files_unchanged']=all(hashlib.sha256((live/n).read_bytes()).hexdigest()==digest for n,digest in json.load(open(P/'production-before-hashes.json')).items() if n not in changed)
code="""from pathlib import Path
import json
r=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=p.joinpath('cmdline').read_bytes().split(b'\\0')
  if b'fnx.server' in cmd or b'fnx.pipeline_worker' in cmd:r.append(dict(pid=int(p.name),cwd=str(p.joinpath('cwd').resolve())))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
assert len(r)==2 and all(x['cwd']=='/root/ascend-dev/fnx' for x in r),r
print(json.dumps(r))
"""
models=json.loads(out('docker','exec','bole-model-Flash','/root/ascend-dev/venv/bin/python','-c',code))
checks['only_canonical_model_pair'] = len(models)==2
r=dict(all_passed=all(checks.values()),checks=checks,temporary_unit_states=states,watchdogs=watchdogs,models=models)
(P/'final-operations.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True)
assert r['all_passed']
