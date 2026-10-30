#!/bin/bash
# ★ B=2 等价性：批维改造后，单序列（B=1 语义）必须仍然给出 2.1756
#   做法：用 FNX_BATCH=2 起服务（单序列走 step_pad，把 token 复制到两行），跑 /fnx/loss
set -u
LOG=/root/ascend-dev/logs/server.log
STAMP=/root/ascend-dev/logs/last_restart
B=${1:-2}

echo "== 语法/导入检查 =="
docker exec fnx bash -lc "cd /root/ascend-dev/fnx && /root/ascend-dev/venv/bin/python -c \
  'import ast;[ast.parse(open(\"fnx/\"+f,encoding=\"utf-8\").read()) for f in (\"engine.py\",\"model.py\",\"server.py\")];print(\"AST OK\")'" 2>&1 | sed 's/^/  /'

echo
echo "== 停服务 =="
systemctl stop fnx.service
pkill -f watchdog.sh 2>/dev/null || true
docker exec fnx bash -lc "pkill -f '[p]ython -m fnx.server' || true"
for i in $(seq 1 30); do
  docker exec fnx bash -lc "pgrep -f '[p]ython -m fnx.server'" >/dev/null 2>&1 || break
  sleep 2
done

echo
echo "== 以 FNX_BATCH=$B 启动（手动，便于看日志）=="
rm -f /root/ascend-dev/logs/batch.log
docker exec -d fnx bash -lc "cd /root/ascend-dev/fnx && \
  FNX_PACK=0 FNX_EXP_G=64 FNX_MOE_KERNEL=0 FNX_NO_PLE=1 FNX_PACKED_TORCH=1 \
  FNX_LPC=46 FNX_NLAYERS=48 FNX_PORT=8000 FNX_BATCH=$B \
  FNX_PREFIX_CACHE=1 FNX_CONV_CACHE=4 FNX_PROF=0 FNX_ENABLE_THINKING=0 \
  /root/ascend-dev/venv/bin/python -m fnx.server > /root/ascend-dev/logs/batch.log 2>&1"
H=""
for i in $(seq 1 30); do
  sleep 10
  H=$(curl -s -m 4 http://127.0.0.1:8000/health || true)
  [ -n "$H" ] && { echo "  ready $H"; break; }
  grep -qiE "out of memory|Tried to allocate|Error|Traceback" /root/ascend-dev/logs/batch.log 2>/dev/null && {
    echo "  !! 启动失败"; grep -vE "^\[W[0-9]" /root/ascend-dev/logs/batch.log | tail -20 | sed 's/^/    /'; exit 1; }
done
[ -z "$H" ] && { echo "  !! 未就绪"; tail -20 /root/ascend-dev/logs/batch.log; exit 1; }
grep -E "hbm[01]:" /root/ascend-dev/logs/batch.log | sed 's/^/  /'

echo
echo "== 等价性：/fnx/loss（B=1 基线 2.1756 / 49.00%）=="
python3 - <<'PY'
import json, urllib.request
text = open('/root/ascend-dev/eval_text.txt', encoding='utf-8').read()
r = urllib.request.Request('http://127.0.0.1:8000/fnx/loss',
    data=json.dumps({"text": text, "max_tokens": 200}).encode(),
    headers={'Content-Type': 'application/json'})
d = json.load(urllib.request.urlopen(r, timeout=900))
ok = abs(d['mean_nll'] - 2.1756) < 0.02
print(f"  n={d['n']} mean_nll={d['mean_nll']:.4f} ppl={d['ppl']:.2f} "
      f"top1={d['top1_acc']*100:.2f}% rank={d['median_rank']}")
print(f"  ⇒ {'等价（差 < 0.02 nats）' if ok else '**不一致，需要排查**'}")
PY

echo
echo "== 单请求仍可用 =="
curl -s -m 120 -X POST http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"你好"}],"max_tokens":16}' \
  | python3 -c "import json,sys;d=json.load(sys.stdin);print('  A:',repr(d['choices'][0]['message']['content']),d['fnx_stats'])"
