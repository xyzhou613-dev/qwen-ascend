#!/bin/bash
set -u
cd /models/fnx-diagnostics/r155-smooth-tpot-ab-20261113
end=$(awk '{print int($1)+2700}' /proc/uptime)
while [ -f lease.active ];do
 now=$(awk '{print int($1)}' /proc/uptime)
 if ! systemctl is-active --quiet fnx-r155-ab-resume || [ "$now" -ge "$end" ];then
  bash "$PWD/rescue.sh" >> safety-restore.log 2>&1 || true
 fi
 [ -f lease.active ] || break
 sleep 20
done
