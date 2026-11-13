#!/bin/bash
set -u
cd /models/fnx-diagnostics/r157-combined-quality-20261113
while [ -f lease.active ];do
 if ! systemctl is-active --quiet "$(cat current-run-unit)" || [ $(( $(date +%s)-$(cat window-start.txt) )) -ge 20100 ];then
  bash "$PWD/rescue.sh" >> safety-restore.log 2>&1 || true
 fi
 [ -f lease.active ] || break
 sleep 20
done
