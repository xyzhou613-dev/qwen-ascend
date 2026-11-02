#!/bin/bash
set -euo pipefail
ROOT=/root/ascend-dev/r104
CANN=/usr/local/Ascend/ascend-toolkit/latest
LT="$CANN/aarch64-linux"
mkdir -p "$ROOT"
"$CANN/bin/bisheng" -shared -fPIC -xcce --cce-aicore-arch=dav-c220 -std=c++17 -O2 \
 -I"$LT/asc" -I"$LT/asc/include" -I"$LT/asc/include/basic_api" -I"$LT/asc/include/adv_api" -I"$LT/include" \
 "$ROOT/r104_combine.cpp" -o "$ROOT/libfnx_combine.so" \
 -L"$LT/lib64" -lascendcl -lruntime -lstdc++ -lgcc_s -lpthread -lm
echo R104_BUILD_ACCEPT
