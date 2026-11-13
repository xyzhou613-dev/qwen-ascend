#!/bin/bash
set -euo pipefail
P=/models/fnx-diagnostics/r156-moe-kmajor-20261113
for d in 0 1;do
 timeout 570 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/launch.py" "$d" layer_probe.py "$d" > "$P/layers-device$d.log" 2>&1
done
