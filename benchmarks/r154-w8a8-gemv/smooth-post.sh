#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r154-w8a8-gemv-20261113
for attempt in $(seq 1 240);do if ! systemctl is-active --quiet fnx-r154-smooth;then break;fi;sleep 2;done
if systemctl is-active --quiet fnx-r154-smooth;then exit 92;fi
for d in 0 1;do docker exec bole-model-Flash /usr/bin/timeout -k 10 680 /root/ascend-dev/venv/bin/python "$PWD/launch.py" "$d" layer_smooth_probe.py "$d" > "layer-smooth-device$d.log" 2>&1;done
for d in 0 1;do docker exec bole-model-Flash /usr/bin/timeout -k 10 680 /root/ascend-dev/venv/bin/python "$PWD/launch.py" "$d" projection_smooth_probe.py "$d" > "projection-smooth-device$d.log" 2>&1;done
