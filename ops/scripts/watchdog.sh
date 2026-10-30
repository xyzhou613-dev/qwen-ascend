#!/bin/bash
# ★ 服务看护（第30轮工程化）：每 20 s 探 /health，连续 3 次失败就用 p275.sh 重启。
#   调用 p275.sh 时带 WD=0（不递归拉看护），并加 mkdir 锁防止与手工重启打架。
# 用法：nohup bash /root/ascend-dev/watchdog.sh > logs/watchdog.log 2>&1 &
set -u
LOG=/root/ascend-dev/logs/watchdog.log
INTERVAL=${WD_INTERVAL:-20}
FAILS=${WD_FAILS:-3}
LOCK=/tmp/fnx_restart.lock
STAMP=/root/ascend-dev/logs/last_restart
GRACE=${WD_GRACE:-150}       # 重启后 150 s 内不改判（装配+图捕获+前缀缓存约 30 s，留余量）
ALIVE_MAX=${WD_ALIVE_MAX:-15}  # 进程还在却 /health 不通时，要连续失败这么多次才重启
fail=0
echo "[$(date '+%F %T')] watchdog 启动 pid=$$ interval=${INTERVAL}s fails=${FAILS} grace=${GRACE}s" >> "$LOG"
while true; do
  now=$(date +%s)
  last=$(cat "$STAMP" 2>/dev/null || echo 0)
  if [ $((now - last)) -lt "$GRACE" ]; then
    # ★ 宽限期：刚重启过就不做判定（否则会在装配窗口内误判 → 与重启互相 pkill 抖动）
    [ "$fail" -gt 0 ] && echo "[$(date '+%F %T')] 宽限期内，清零失败计数" >> "$LOG"
    fail=0
    sleep "$INTERVAL"
    continue
  fi
  H=$(curl -s -m 6 http://127.0.0.1:8000/health 2>/dev/null || true)
  if echo "$H" | grep -q '"status":"ok"'; then
    [ "$fail" -gt 0 ] && echo "[$(date '+%F %T')] 恢复（之前失败 $fail 次）" >> "$LOG"
    fail=0
  else
    fail=$((fail+1))
    echo "[$(date '+%F %T')] health 失败 $fail/$FAILS: ${H:0:60}" >> "$LOG"
    # ★ 进程还在 + /health 不通 = 可能是长时间生成把事件循环堵住了（旧 bug），
    #   这种情况**不要急着重启**（会把正在跑的请求杀掉），要连续 ALIVE_MAX 次才动。
    ALIVE=$(docker exec fnx bash -lc "pgrep -f '[p]ython -m fnx.server'" >/dev/null 2>&1 && echo 1 || echo 0)
    LIM=$FAILS
    [ "$ALIVE" = 1 ] && LIM=$ALIVE_MAX
    if [ "$fail" -ge "$LIM" ]; then
      if mkdir "$LOCK" 2>/dev/null; then
        echo "[$(date '+%F %T')] → 触发重启" >> "$LOG"
        WD=0 bash /root/ascend-dev/p275.sh >> "$LOG" 2>&1
        rmdir "$LOCK" 2>/dev/null || true
      else
        echo "[$(date '+%F %T')] 已有重启在进行，跳过（lock 存在）" >> "$LOG"
      fi
      fail=0
    fi
  fi
  sleep "$INTERVAL"
done
