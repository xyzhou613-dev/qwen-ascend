#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r153-tpot-directions-20261113
for d in 0 1;do timeout 530 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" "$d" verify_layer_loop_probe.py "$d" > "verify-layer-loop-device$d.log" 2>&1;done
