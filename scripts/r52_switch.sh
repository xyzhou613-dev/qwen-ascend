#!/bin/bash
set -euo pipefail
case "${1:-}" in
  8k|8192) ctx=8192;lpc=24;;
  32k|32768) ctx=32768;lpc=24;;
  64k|65536) ctx=65536;lpc=24;;
  128k|131072) ctx=131072;lpc=24;;
  256k|262144) ctx=262144;lpc=24;;
  *) echo 'Usage: r52_switch.sh {8k|32k|64k|128k|256k}';exit 2;;
esac
printf 'CTX=%s\nLPC=%s\n' "$ctx" "$lpc" > /root/ascend-dev/r52/active_profile.env
systemctl restart fnx.service
curl -fsS --max-time 5 http://127.0.0.1:8000/health
echo
