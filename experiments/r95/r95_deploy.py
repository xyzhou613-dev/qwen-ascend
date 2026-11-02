from pathlib import Path
import json,shutil,statistics,subprocess,tarfile
r=Path('/root/ascend-dev');s=r/'r95';repo=Path('/root/qwen-ascend')
assert 'R95_CANDIDATE_ACCEPT' in (r/'logs/r95_run.log').read_text()
assert 'R95_STATE_ACCEPT' in (r/'logs/r95_run.log').read_text()
assert 'R94_INDEX_ACCEPT' in (r/'logs/r94_index_final.log').read_text()
old=[json.loads((r/'r92'/name/'result_262144.json').read_text()) for name in ('production256','production256_second')]
new=[json.loads((s/name/'result_262144.json').read_text()) for name in ('long1','long2')]
for a in new:
    assert a['recall_hits']==3 and a['stats']['n_prefill']==262012 and a['stats']['tokens']==128
    assert a['ttft_s']<120 and a['stats']['tpot_ms']<1000/15
for name in ('long1','long2'):
    c=json.loads((s/name/'continuation.json').read_text())
    assert c['stats']['n_prefill']==29 and c['answer'].strip()=='7319'
assert statistics.mean(a['ttft_s'] for a in new)<statistics.mean(a['ttft_s'] for a in old)*1.01
assert statistics.mean(a['stats']['tpot_ms'] for a in new)<statistics.mean(a['stats']['tpot_ms'] for a in old)*1.01
short=json.loads((s/'short/results.json').read_text())
assert statistics.median(a['ttft_s'] for a in short)<5.874*1.02
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo)
assert not subprocess.check_output(['git','diff','--name-only','00a181fb','HEAD','--','fnx','scripts/prod_pipeline.sh'],cwd=repo)
backup=s/'predeploy.tgz';assert not backup.exists()
with tarfile.open(backup,'w:gz') as tf:
    for name in ('fnx/fnx','p275.sh','r52/active_profile.env'):tf.add(r/name,arcname=name)
rollback='''#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev
test -f "$ROOT/r95/predeploy.tgz"
systemctl stop fnx.service
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
tar -xzf "$ROOT/r95/predeploy.tgz" -C "$ROOT"
rm -f "$ROOT/fnx/fnx/snapshot.py"
systemctl start fnx.service
curl -fsS --max-time 5 http://127.0.0.1:8000/health
'''
(r/'r95_rollback.sh').write_text(rollback)
subprocess.check_call(['bash','-n',str(r/'r95_rollback.sh')])
subprocess.check_call(['systemctl','stop','fnx.service'])
shutil.copy2(r/'logs/pipeline_worker.log',s/'candidate-worker.log')
subprocess.check_call(['docker','exec','fnx','bash','-lc',"pkill -f '[p]ython -m fnx.server' || true"])
try:
    # Explore the next trace hotspot while devices are exclusively available.
    # This probe does not modify candidate or production code.
    shutil.copy2(r/'r96_write_probe.py',s/'write_probe.py')
    with (r/'logs/r96_write_probe.log').open('w') as log:
        subprocess.check_call(['docker','exec','fnx','bash','-lc','cd /root/ascend-dev/r95; ASCEND_RT_VISIBLE_DEVICES=0 /root/ascend-dev/venv/bin/python write_probe.py'],stdout=log,stderr=subprocess.STDOUT)
    # Add read-only health fields; computation is the already-tested candidate.
    p=s/'fnx/server.py';text=p.read_text();needle='            "native_dual": os.environ.get("FNX_NATIVE_DUAL", "0") == "1",'
    assert text.count(needle)==1
    if '"conversation_state_bytes"' not in text:text=text.replace(needle,needle+'''
            "index_unique": os.environ.get("FNX_INDEX_UNIQUE", "0") == "1",
            "state_dual_slim": os.environ.get("FNX_STATE_DUAL_SLIM", "0") == "1",
            "conversation_state_bytes": sum(model.state_bytes(st) for st in getattr(STATE.get("e"), "conv_slots", [])),''')
    p.write_text(text);compile(text,str(p),'exec')
    for name in ('engine.py','model.py','snapshot.py','server.py'):shutil.copy2(s/'fnx'/name,r/'fnx/fnx'/name)
    p=r/'p275.sh';text=p.read_text();needle='  /root/ascend-dev/venv/bin/python -m fnx.server'
    assert text.count(needle)==1 and 'FNX_STATE_DUAL_SLIM=' not in text
    text=text.replace(needle,'  FNX_INDEX_UNIQUE=1 FNX_STATE_DUAL_SLIM=1 \\\n'+needle).replace('# r92:','# r95:')
    p.write_text(text);subprocess.check_call(['bash','-n',str(p)])
    subprocess.check_call(['systemctl','start','fnx.service'])
except BaseException:
    subprocess.call(['bash',str(r/'r95_rollback.sh')]);raise
print('R95_DEPLOYED',flush=True)
