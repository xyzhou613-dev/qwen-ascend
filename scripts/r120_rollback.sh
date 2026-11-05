#!/bin/bash
set -euo pipefail
systemctl stop fnx.service
docker exec fnx pkill -TERM -f '[p]ython -m fnx.server' || true
tar -xzf /models/fnx-diagnostics/r120-service-20261105/predeploy-server.tgz -C /root/ascend-dev
rm -f /root/ascend-dev/fnx/fnx/request_validation.py
systemctl start fnx.service
