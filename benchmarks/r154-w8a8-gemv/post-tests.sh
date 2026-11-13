#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r154-w8a8-gemv-20261113
for attempt in $(seq 1 180);do if ! systemctl is-active --quiet fnx-r154-sweep-selective;then break;fi;sleep 2;done
if systemctl is-active --quiet fnx-r154-sweep-selective;then exit 92;fi
docker exec bole-model-Flash /usr/bin/timeout -k 10 220 /root/ascend-dev/venv/bin/python "$PWD/launch.py" 0 transpose_probe.py > transpose.log 2>&1
for d in 0 1;do docker exec bole-model-Flash /usr/bin/timeout -k 10 680 /root/ascend-dev/venv/bin/python "$PWD/launch.py" "$d" layer_selective_probe.py "$d" > "layer-selective-device$d.log" 2>&1;done
for d in 0 1;do docker exec bole-model-Flash /usr/bin/timeout -k 10 680 /root/ascend-dev/venv/bin/python "$PWD/launch.py" "$d" projection_probe.py "$d" > "projection-device$d.log" 2>&1;done
