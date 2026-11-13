#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r154-w8a8-gemv-20261113
for mode in reference selected;do docker exec bole-model-Flash /usr/bin/timeout -k 10 240 /root/ascend-dev/venv/bin/python "$PWD/launch_selected.py" 0 loader_check.py "$mode" > "loader-$mode.log" 2>&1;done
