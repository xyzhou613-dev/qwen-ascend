#!/bin/bash
# Build inside the production CANN environment; generated .so is not source.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
OUTPUT=${1:-"$HERE/libtiled.so"}
CANN=${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
LT="$CANN/aarch64-linux"
mkdir -p -- "$(dirname -- "$OUTPUT")"
"$CANN/bin/bisheng" -shared -fPIC -xcce --cce-aicore-arch=dav-c220-vec -std=c++17 -O2 \
 -I"$LT/asc" -I"$LT/asc/include" -I"$LT/asc/include/basic_api" -I"$LT/asc/include/adv_api" -I"$LT/include" \
 "$HERE/delta.cpp" -o "$OUTPUT" \
 -L"$LT/lib64" -lascendcl -lruntime -lstdc++ -lgcc_s -lpthread -lm
echo "Built $OUTPUT"
