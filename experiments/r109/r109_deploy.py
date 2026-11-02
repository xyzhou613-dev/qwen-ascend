from pathlib import Path
import hashlib,json,shutil,statistics,subprocess,tarfile
r=Path('/root/ascend-dev');s=r/'r109';repo=Path('/root/qwen-ascend')
assert 'R109_CANDIDATE_ACCEPT' in (r/'logs/r109_run.log').read_text()
assert 'R109_DELTA_ACCEPT' in (r/'logs/r109_run.log').read_text()
probe=json.loads((r/'r108/probe.json').read_text());assert all(x['state_exact'] and x['output_exact'] and x['recurrent_exact_256'] and x['signed_zero_exact'] for x in probe)
a=json.loads((s/'long/result_262144.json').read_text());base=json.loads((r/'r106/summary.json').read_text())['production256']
assert a['recall_hits']==3 and a['stats']['tokens']==128 and a['stats']['n_prefill']==262012
assert a['ttft_s']<statistics.mean(x['ttft_s'] for x in base)*1.01
assert a['stats']['tpot_ms']<statistics.mean(x['stats']['tpot_ms'] for x in base)-.5
c=json.loads((s/'long/continuation.json').read_text());assert c['stats']['n_prefill']==29 and c['answer'].strip()=='7319'
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo)
assert not subprocess.check_output(['git','diff','--name-only','390691a','HEAD','--','fnx','scripts/prod_pipeline.sh'],cwd=repo)
backup=s/'predeploy.tgz';assert not backup.exists()
with tarfile.open(backup,'w:gz') as tf:
    for name in ('fnx/fnx','p275.sh','r52/active_profile.env'):tf.add(r/name,arcname=name)
rollback='''#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev
test -f "$ROOT/r109/predeploy.tgz"
systemctl stop fnx.service
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
tar -xzf "$ROOT/r109/predeploy.tgz" -C "$ROOT"
rm -f "$ROOT/fnx/fnx/delta_native.py" "$ROOT/fnx/fnx/libfnx_delta.so"
systemctl start fnx.service
curl -fsS --max-time 5 http://127.0.0.1:8000/health
'''
(r/'r109_rollback.sh').write_text(rollback);subprocess.check_call(['bash','-n',str(r/'r109_rollback.sh')])
subprocess.check_call(['systemctl','stop','fnx.service']);shutil.copy2(r/'logs/pipeline_worker.log',s/'candidate-worker.log')
subprocess.check_call(['docker','exec','fnx','bash','-lc',"pkill -f '[p]ython -m fnx.server' || true"])
try:
    # Rebuild the archived source to check the build recipe, retain hashes,
    # and deploy the candidate binary used for all acceptance requests.
    build=s/'build';build.mkdir(exist_ok=True)
    shutil.copy2(r/'r108/r108_delta.cpp',build/'delta.cpp');shutil.copy2(r/'r109_build_delta.sh',build/'build.sh')
    with (r/'logs/r109_production_build.log').open('w') as log:
        subprocess.check_call(['docker','exec','fnx','bash',str(build/'build.sh'),str(build/'libfnx_delta.so')],stdout=log,stderr=subprocess.STDOUT)
    candidate_hash=hashlib.sha256((s/'fnx/libfnx_delta.so').read_bytes()).hexdigest()
    rebuilt_hash=hashlib.sha256((build/'libfnx_delta.so').read_bytes()).hexdigest()
    # Compiler embeds source-path metadata, so equal binary hashes are not a
    # portability guarantee. Deploy the already-tested binary and retain both.
    (s/'binary-manifest.json').write_text(json.dumps({'candidate_sha256':candidate_hash,'rebuilt_sha256':rebuilt_hash,
        'source_sha256':hashlib.sha256((build/'delta.cpp').read_bytes()).hexdigest(),'arch':'dav-c220-vec'},indent=2))
    for name in ('engine.py','server.py','delta_native.py','libfnx_delta.so'):shutil.copy2(s/'fnx'/name,r/'fnx/fnx'/name)
    p=r/'p275.sh';text=p.read_text();needle='  /root/ascend-dev/venv/bin/python -m fnx.server'
    assert text.count(needle)==1 and 'FNX_DN_NATIVE=' not in text
    text=text.replace(needle,'  FNX_DN_NATIVE=1 \\\n'+needle).replace('# r106:','# r109:')
    p.write_text(text);subprocess.check_call(['bash','-n',str(p)]);subprocess.check_call(['systemctl','start','fnx.service'])
except BaseException:
    subprocess.call(['bash',str(r/'r109_rollback.sh')]);raise
print('R109_DEPLOYED',flush=True)
