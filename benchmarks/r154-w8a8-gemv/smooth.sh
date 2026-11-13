#!/bin/bash
set -euo pipefail
cd /models/fnx-diagnostics/r154-w8a8-gemv-20261113
docker exec bole-model-Flash /usr/bin/timeout -k 10 90 /root/ascend-dev/venv/bin/python "$PWD/prepare_calibration.py" > prepare-calibration.log 2>&1
docker exec bole-model-Flash /usr/bin/timeout -k 10 1000 /root/ascend-dev/venv/bin/python "$PWD/launch.py" 0 calibrate.py > calibrate.log 2>&1
docker exec bole-model-Flash /usr/bin/timeout -k 10 2400 /root/ascend-dev/venv/bin/python "$PWD/launch.py" 0 sweep_smooth.py > sweep-smooth.log 2>&1
