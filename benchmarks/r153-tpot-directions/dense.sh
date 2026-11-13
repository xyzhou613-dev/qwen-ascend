#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r153-tpot-directions-20261113
for d in 0 1;do timeout 520 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" "$d" dense_probe.py "$d" > "dense-device$d.log" 2>&1;done
