#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r154-w8a8-gemv-20261113
timeout 180 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/prepare_gemv.py" > gemv-repack.log 2>&1
for d in 0 1;do timeout 680 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" "$d" gemv_probe.py "$d" > "gemv-device$d.log" 2>&1;done
