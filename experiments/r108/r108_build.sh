#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev/r108
CANN=/usr/local/Ascend/ascend-toolkit/latest
LT="$CANN/aarch64-linux"
"$CANN/bin/bisheng" -shared -fPIC -xcce --cce-aicore-arch=dav-c220-vec -std=c++17 -O2 \
 -I"$LT/asc" -I"$LT/asc/include" -I"$LT/asc/include/basic_api" -I"$LT/asc/include/adv_api" -I"$LT/include" \
 "$ROOT/r108_delta.cpp" -o "$ROOT/libfnx_delta.so" \
 -L"$LT/lib64" -lascendcl -lruntime -lstdc++ -lgcc_s -lpthread -lm
echo R108_BUILD_ACCEPT
