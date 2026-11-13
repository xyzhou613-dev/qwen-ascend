#!/bin/bash
set -euo pipefail
P=/models/fnx-diagnostics/r156-moe-kmajor-20261113
for d in 0 1;do
 timeout 520 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/launch.py" "$d" merge_probe.py "$d" > "$P/merge-device$d.log" 2>&1
done
