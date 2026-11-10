#!/bin/bash
set -euo pipefail
for visible in 0 1;do
 docker exec bole-model-Flash bash -lc "cd /models/fnx-diagnostics/r136-decode-budget-20261110;ASCEND_RT_VISIBLE_DEVICES=$visible OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 /root/ascend-dev/venv/bin/python calibrate_single.py > calibration-$visible.log 2>&1"
done
