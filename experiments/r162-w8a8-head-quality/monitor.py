"""Read-only compact progress; never print model answers or private inputs."""
import json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent
def read(name):
    p=P/name
    try:return json.loads(p.read_text())
    except (FileNotFoundError,json.JSONDecodeError):return None
def state(unit):return subprocess.check_output(['systemctl','show',unit,'-p','ActiveState','--value'],text=True).strip()
def count(name):
    p=P/name
    if not p.exists():return 0
    return sum(1 for x in p.read_bytes().splitlines(keepends=True) if x.endswith(b'\n'))
out=dict(time=time.strftime('%H:%M:%S'),elapsed_s=int(time.time()-int((P/'window-start.txt').read_text())),
    main=state('fnx-r162-run.service'),rescue=state('fnx-r162-rescue.service'),lease=(P/'lease.active').exists())
out['ready']=[p for p in ('rollback','smooth','combined') if (P/f'{p}-startup.json').exists()]
out['http']={}
for phase in ('rollback','smooth','combined'):
    summary=read(phase+'/summary.json')
    if summary:out['http'][phase]={k:summary[k] for k in ('median_8k_tpot_ms','median_256k_tpot_ms','sustained1024_tpot_ms')}
    else:
        completed={}
        for name in ('8k.json','256k.json','cached256k.json','sustained1024.json'):
            data=read(phase+'/'+name)
            if data is not None:completed[name]=len(data) if isinstance(data,list) else 1
        if completed:out['http'][phase]=completed
out['quality_counts']={phase:dict(tasks=count(f'quality-{phase}.jsonl'),nll=count(f'{phase}-nll.jsonl'),natural=count(f'{phase}-natural.jsonl')) for phase in ('rollback','combined')}
out['baseline_reuse_verified']=(P/'baseline-reuse-repeat.json').exists()
gate=read('short-gate.json')
if gate:out['short_gate']=gate
restore=read('restore-complete.json')
if restore:out['restoration']=restore
lines=(P/'controller.log').read_text(errors='replace').splitlines()
lines=[x for x in lines if not x.startswith('curl:')]
out['last_event']=lines[-1][:900] if lines else ''
if any('Traceback (most recent call last)' in x for x in lines[-12:]):out['controller_error_tail']=lines[-10:]
print(json.dumps(out,separators=(',',':')),flush=True)
