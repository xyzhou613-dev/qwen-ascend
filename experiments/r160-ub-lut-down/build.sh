#!/bin/bash
set -euo pipefail
P=/models/fnx-diagnostics/r160-ub-lut-down-20261114
CANN=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
LT="$CANN/aarch64-linux"
"$CANN/bin/bisheng" -shared -fPIC -xcce --cce-aicore-arch=dav-c220-vec -std=c++17 -O2 \
 -I"$LT/asc" -I"$LT/asc/include" -I"$LT/asc/include/basic_api" -I"$LT/asc/include/adv_api" -I"$LT/include" \
 "$P/lut_down.cpp" -o "$P/libfnx_w4_lut.so" \
 -L"$LT/lib64" -lascendcl -lruntime -lstdc++ -lgcc_s -lpthread -lm
