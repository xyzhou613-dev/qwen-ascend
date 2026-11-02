from pathlib import Path
import json,shutil,statistics,subprocess,tarfile
r=Path('/root/ascend-dev');s=r/'r100';repo=Path('/root/qwen-ascend')
assert 'R100_CANDIDATE_ACCEPT' in (r/'logs/r100_run.log').read_text()
assert 'R100_MOE_ACCEPT' in (r/'logs/r100_run.log').read_text()
a=json.loads((s/'long/result_262144.json').read_text())
base=json.loads((r/'r95/summary.json').read_text())['production256']
assert a['recall_hits']==3 and a['stats']['tokens']==128 and a['stats']['n_prefill']==262012
assert a['ttft_s']<statistics.mean(x['ttft_s'] for x in base)-.5
assert a['stats']['tpot_ms']<statistics.mean(x['stats']['tpot_ms'] for x in base)*1.01
c=json.loads((s/'long/continuation.json').read_text());assert c['stats']['n_prefill']==29 and c['answer'].strip()=='7319'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo)
assert not subprocess.check_output(['git','diff','--name-only','91ba026','HEAD','--','fnx','scripts/prod_pipeline.sh'],cwd=repo)
backup=s/'predeploy.tgz';assert not backup.exists()
with tarfile.open(backup,'w:gz') as tf:
    for name in ('fnx/fnx','p275.sh','r52/active_profile.env'):tf.add(r/name,arcname=name)
rollback='''#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev
test -f "$ROOT/r100/predeploy.tgz"
systemctl stop fnx.service
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
tar -xzf "$ROOT/r100/predeploy.tgz" -C "$ROOT"
systemctl start fnx.service
curl -fsS --max-time 5 http://127.0.0.1:8000/health
'''
(r/'r100_rollback.sh').write_text(rollback);subprocess.check_call(['bash','-n',str(r/'r100_rollback.sh')])
subprocess.check_call(['systemctl','stop','fnx.service'])
shutil.copy2(r/'logs/pipeline_worker.log',s/'candidate-worker.log')
subprocess.check_call(['docker','exec','fnx','bash','-lc',"pkill -f '[p]ython -m fnx.server' || true"])
try:
    for name in ('engine.py','server.py'):shutil.copy2(s/'fnx'/name,r/'fnx/fnx'/name)
    p=r/'p275.sh';text=p.read_text();needle='  /root/ascend-dev/venv/bin/python -m fnx.server'
    assert text.count(needle)==1 and 'FNX_MOE_INVERSE=' not in text
    text=text.replace(needle,'  FNX_MOE_INVERSE=1 \\\n'+needle).replace('# r95:','# r100:')
    p.write_text(text);subprocess.check_call(['bash','-n',str(p)])
    subprocess.check_call(['systemctl','start','fnx.service'])
except BaseException:
    subprocess.call(['bash',str(r/'r100_rollback.sh')]);raise
print('R100_DEPLOYED',flush=True)
