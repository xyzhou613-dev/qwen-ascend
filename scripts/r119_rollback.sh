#!/bin/bash
set -euo pipefail
cd /root/ascend-dev
systemctl stop fnx.service
docker exec fnx pkill -TERM -f '[p]ython -m fnx.server' || true
tar -xzf /root/ascend-dev/r119/predeploy.tgz
rm -f fnx/fnx/weighted_reduce.py fnx/fnx/libfnx_reduce.so
systemctl start fnx.service
