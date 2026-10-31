#!/bin/bash
set -euo pipefail
systemctl stop fnx.service || true
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
sleep 2
tar -xzf /root/ascend-dev/r55/predeploy.tgz -C /root/ascend-dev
# The user requested a persistent 256k deployment, including on rollback.
printf 'CTX=262144\nLPC=38\n' > /root/ascend-dev/r52/active_profile.env
systemctl start fnx.service
curl -fsS --max-time 10 http://127.0.0.1:8000/health
echo
