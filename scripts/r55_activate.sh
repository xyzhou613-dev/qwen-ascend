#!/bin/bash
set -euo pipefail
python3 - <<'PY'
import json
from pathlib import Path
r=json.loads(Path('/root/ascend-dev/r55/layout/result_262144.json').read_text())
assert r['prompt_tokens']==262016 and r['stats']['n_prefill']==262012
assert r['recall_hits']==3 and r['ttft_s']<=120 and r['stats']['tpot_ms']<=1000/15
assert 'LONG_ACCEPT 262144' in Path('/root/ascend-dev/logs/r55_layout256.log').read_text()
print('256k TTFT, decode speed, recall, and cache acceptance passed')
PY
test -f /root/ascend-dev/r55/predeploy.tgz || tar -czf /root/ascend-dev/r55/predeploy.tgz -C /root/ascend-dev fnx/fnx p275.sh watchdog.sh r52/active_profile.env r52_switch.sh
rollback() {
 trap - ERR
 bash /root/ascend-dev/r55_rollback.sh
}
trap rollback ERR
systemctl stop fnx.service
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
sleep 3
for name in engine model server pipeline pipeline_worker;do cp /root/ascend-dev/r55/fnx/$name.py /root/ascend-dev/fnx/fnx/$name.py;done
python3 - <<'PY'
from pathlib import Path
p=Path('/root/ascend-dev/p275.sh');s=p.read_text()
changes={
 'FNX_PACK=0':'ASCEND_RT_VISIBLE_DEVICES=0 FNX_PIPELINE=1 FNX_PIPELINE_DEVICE1=1 FNX_PIPELINE_SPLIT=$LPC FNX_PACK=0',
 'FNX_CONV_CACHE=4':'FNX_CONV_CACHE=1 FNX_CACHE_GPU=1',
 'FNX_DEQ_ATTN=0':'FNX_DEQ_ATTN=48',
 'FNX_CHUNK=512':'FNX_CHUNK=4096 FNX_CHUNK_SIZES=4096,512,128 FNX_CHUNK_POOL=private',
 'FNX_QSA_SPARSE=1':'FNX_QSA_SPARSE=1 FNX_DELTA_BATCH=1 FNX_INDEX_CACHE=1 FNX_INDEX_TILE=512 FNX_QSA_TILE=128 FNX_KV_POSITION_MAJOR=1 FNX_SPARSE_FLASH=1 FNX_DN_DECODE_BMM=0 FNX_GRAPH_G1=0 FNX_GRAPH_HEAD=0',
 'FNX_MOE_FAST=${MFAST:-0}':'FNX_MOE_FAST=1 FNX_MOE_ROUTER_FAST=0 FNX_MOE_ACT_BF16=0',
}
for old,new in changes.items():
 if new in s:
  assert s.count(new)==1,(new,s.count(new))
  continue
 assert s.count(old)==1,(old,s.count(old))
 s=s.replace(old,new,1)
p.write_text(s)
Path('/root/ascend-dev/r52/active_profile.env').write_text('CTX=262144\nLPC=24\n')
p=Path('/root/ascend-dev/r52_switch.sh');s=p.read_text()
for old in ('lpc=42','lpc=41','lpc=40','lpc=38'):
 s=s.replace(old,'lpc=24')
p.write_text(s)
PY
bash -n /root/ascend-dev/p275.sh
bash -n /root/ascend-dev/r52_switch.sh
systemctl start fnx.service
systemctl is-active fnx.service
systemctl is-enabled fnx.service
curl -fsS --max-time 10 http://127.0.0.1:8000/health
echo
echo PRODUCTION_256K_ACTIVE
