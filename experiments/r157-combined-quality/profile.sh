#!/bin/bash
set -euo pipefail
P=/models/fnx-diagnostics/r157-combined-quality-20261113
for phase in combined smooth;do
 elapsed=$(( $(date +%s)-$(cat "$P/window-start.txt") ))
 remaining=$((20100-elapsed))
 if [ "$remaining" -lt 1800 ];then echo PROFILE_SKIPPED_TIME_BUDGET "$phase";break;fi
 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/stop_owned.py"
 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/launch_profile.py" "$phase"
 for count in $(seq 1 550);do
  sleep 2
  if grep -q R145_PROFILE_COMPLETE "$P/profile-$phase/profile.log";then break;fi
  if grep -qE 'Traceback \(most recent call last\)|ERR99999|out of memory' "$P/profile-$phase/profile.log";then echo PROFILE_FAILED "$phase";tail -n 16 "$P/profile-$phase/profile.log";exit 3;fi
 done
 grep -q R145_PROFILE_COMPLETE "$P/profile-$phase/profile.log"
 docker exec bole-model-Flash /root/ascend-dev/venv/bin/python "$P/stop_owned.py"
done
