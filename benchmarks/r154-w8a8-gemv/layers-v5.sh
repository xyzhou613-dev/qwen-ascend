#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r154-w8a8-gemv-20261113
for d in 0 1;do timeout 680 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$PWD/launch.py" "$d" layer_bounded_probe.py "$d" > "layer-v5-device$d.log" 2>&1;done
