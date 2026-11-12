#!/bin/bash
# Keep recovery monitoring outside the experiment controller's cgroup.
set -euo pipefail
systemctl stop fnx-watchdog-guard.service 2>/dev/null || true
if pgrep -f '^bash /root/ascend-dev/watchdog.sh$' > /tmp/fnx-r144-watchdog.pids;then
 while read -r pid;do kill -TERM "$pid" 2>/dev/null || true;done < /tmp/fnx-r144-watchdog.pids
fi
rm -f /tmp/fnx-r144-watchdog.pids
systemd-run --unit=fnx-watchdog-guard --collect /bin/bash /root/ascend-dev/watchdog.sh
