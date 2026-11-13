"""Run on the host after canonical restore and HTTP smoke; no private ENV output."""
import hashlib,json,subprocess,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def system(*args):return subprocess.check_output(['systemctl',*args],text=True).strip()
health=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
before=json.load(open(P/'production-before-hashes.json'))
after={name:digest(Path('/root/ascend-dev/fnx/fnx')/name) for name in before}
assert before==after,'PRODUCTION_SOURCE_CHANGED'
p275_hash=digest(Path('/root/ascend-dev/p275.sh'))
assert p275_hash==digest(P/'p275-before.sh'),'CANONICAL_ENTRY_CHANGED'
assert health['service_revision']=='r152' and health['worker_alive'] and health['ctx']==262144
assert not health.get('dense_w8a8_enabled',False) and 'diagnostic_variant' not in health
assert not (P/'lease.active').exists() and not (P/'original-private.json').exists()
assert not Path('/run/systemd/system/model-switch.service.d/r155-exclusive.conf').exists()
assert system('is-active','model-switch.service')=='active'
assert system('show','model-switch.service','--value','-p','RefuseManualStart')=='no'
procs=[];watchdogs=[]
for proc in Path('/proc').glob('[0-9]*'):
 try:
  cmd=[x.decode(errors='replace') for x in proc.joinpath('cmdline').read_bytes().split(b'\0') if x]
  if not cmd:continue
  if 'fnx.server' in cmd or 'fnx.pipeline_worker' in cmd:
   cwd=str(proc.joinpath('cwd').resolve());assert cwd=='/root/ascend-dev/fnx',(proc.name,cwd)
   env=dict(x.split(b'=',1) for x in proc.joinpath('environ').read_bytes().split(b'\0') if b'=' in x)
   assert env.get(b'FNX_DENSE_W8A8',b'0')!=b'1'
   procs.append(dict(pid=int(proc.name),module='fnx.server' if 'fnx.server' in cmd else 'fnx.pipeline_worker',cwd=cwd,
    visible=env.get(b'ASCEND_RT_VISIBLE_DEVICES',b'').decode()))
  if len(cmd)>=2 and cmd[0].endswith('bash') and cmd[1]=='/root/ascend-dev/watchdog.sh':
   ppid=int(proc.joinpath('status').read_text().split('PPid:')[1].splitlines()[0]);watchdogs.append(dict(pid=int(proc.name),ppid=ppid))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
assert sorted(x['module'] for x in procs)==['fnx.pipeline_worker','fnx.server'],procs
assert len(watchdogs)==1 and watchdogs[0]['ppid']==1,watchdogs
units={}
for unit in ('fnx-r155-ab','fnx-r155-ab-resume','fnx-r155-safety','fnx-r155-safety-resume','fnx-r155-rescue.timer','fnx-r155-rescue.service'):
 state=system('show',unit,'--value','-p','ActiveState');units[unit]=state
 assert state not in ('active','activating','reloading'),(unit,state)
smoke=json.load(open(P/'production-http.json'));assert smoke['answer_matches_baseline']
result=dict(timestamp_epoch=time.time(),production_health=health,source_file_count=len(after),production_hashes_unchanged=before==after,
 production_hashes=after,p275_sha256=p275_hash,p275_unchanged=True,production_processes=procs,watchdogs=watchdogs,
 manager_active=True,manager_refuse_manual_start=False,temporary_dropin_removed=True,private_environment_deleted=True,
 lease_deleted=True,experimental_units=units,formal_http_smoke_passed=True,candidate_deployed=False)
(P/'final-operations.json').write_text(json.dumps(result,indent=2));print('FINAL_OPERATIONS_PASS',len(after),procs,watchdogs,flush=True)
