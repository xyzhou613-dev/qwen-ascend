#!/bin/bash
set -u
restore() {
  docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
  systemctl start fnx.service
}
trap restore EXIT
systemctl stop fnx.service
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
sleep 3
for lpc in 40 38; do
  echo "CAPACITY ctx=262144 lpc=$lpc"
  docker exec fnx bash -lc "set -a; source /root/ascend-dev/r53/baseline.env; set +a; export FNX_CTX=262144 FNX_LPC=$lpc; cd /root/ascend-dev/fnx; timeout -k 20s 180s /root/ascend-dev/venv/bin/python tests/t_r52_startup.py" > /root/ascend-dev/logs/r53_capacity_$lpc.log 2>&1
  rc=$?;echo "CAPACITY_EXIT lpc=$lpc rc=$rc";tail -4 /root/ascend-dev/logs/r53_capacity_$lpc.log
  if [ "$rc" = 0 ];then break;fi
done
[ "$rc" = 0 ] || exit 1
printf 'CTX=262144\nLPC=%s\n' "$lpc" > /root/ascend-dev/r53/test_profile.env
docker exec -d fnx bash -lc "set -a; source /root/ascend-dev/r53/baseline.env; set +a; export FNX_CTX=262144 FNX_LPC=$lpc FNX_CONV_CACHE=2; cd /root/ascend-dev/fnx; /root/ascend-dev/venv/bin/python -m fnx.server > /root/ascend-dev/logs/r53_server.log 2>&1"
ready=0
for i in $(seq 1 120); do
  sleep 2
  if curl -fsS --max-time 3 http://127.0.0.1:8000/health > /root/ascend-dev/r53/health.json 2>/dev/null;then ready=1;break;fi
  grep -qiE 'out of memory|Insufficient_Device_Memory|Traceback' /root/ascend-dev/logs/r53_server.log && break
done
[ "$ready" = 1 ] || { tail -10 /root/ascend-dev/logs/r53_server.log;exit 1; }
docker exec fnx bash -lc 'TEST_CTX=262144 /root/ascend-dev/venv/bin/python /root/ascend-dev/r52_long_client.py' > /root/ascend-dev/logs/r53_long.log 2>&1
rc=$?;echo "LONG_EXIT=$rc";tail -5 /root/ascend-dev/logs/r53_long.log
exit "$rc"
