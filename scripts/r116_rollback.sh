#!/bin/bash
set -euo pipefail
cd /root/ascend-dev
systemctl stop fnx.service
docker exec fnx pkill -TERM -f '[p]ython -m fnx.server' || true
tar -xzf /root/ascend-dev/r116/predeploy.tgz
rm -f fnx/fnx/conv_native.py fnx/fnx/libfnx_conv.so fnx/fnx/index_expand.py fnx/fnx/libfnx_index_expand.so
systemctl start fnx.service
