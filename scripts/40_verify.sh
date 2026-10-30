#!/bin/bash
# =============================================================================
# 40 · 验收：把实测值与 BASELINE.md §4 比对，全部在容差内才打印 BASELINE OK
#   用法：  bash 40_verify.sh          # 全量（含 24 题 MCQ，约 4 分钟）
#           bash 40_verify.sh --quick  # 跳过 MCQ
# =============================================================================
set -u
RUN=${RUN:-/root/ascend-dev}
CTN=${CTN:-fnx}
PY="$RUN/venv/bin/python"
QUICK=0
[ "${1:-}" = "--quick" ] && QUICK=1
ECHO=$RUN/eval_text.txt
fail=0
chk() { # chk 名称 实测 期望 判据
  local name=$1 got=$2 exp=$3 rule=$4
  if python3 -c "import sys;a=float('$got');b=float('$exp');sys.exit(0 if $rule else 1)" 2>/dev/null; then
    printf '  \033[32mOK\033[0m   %-34s %s（基线 %s）\n' "$name" "$got" "$exp"
  else
    printf '  \033[31mFAIL\033[0m %-34s %s（基线 %s）\n' "$name" "$got" "$exp"; fail=$((fail+1))
  fi
}

echo "== 40 验收（对照 BASELINE.md，基线 baseline-r31）=="

# ---- 1. 存活 ----
H=$(curl -s -m 5 http://127.0.0.1:8000/health || true)
if echo "$H" | grep -q '"status":"ok"'; then printf '  \033[32mOK\033[0m   %-34s %s\n' "/health" "$H"
else printf '  \033[31mFAIL\033[0m /health → %s\n' "${H:-无响应}"; echo "  其余检查跳过"; exit 1; fi

# ---- 2. 质量口径 ----
LOSS=$(python3 - "$ECHO" <<'PY'
import json, sys, urllib.request
text = open(sys.argv[1], encoding="utf-8").read()
r = urllib.request.Request("http://127.0.0.1:8000/fnx/loss",
    data=json.dumps({"text": text, "max_tokens": 200}).encode(),
    headers={"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(r, timeout=1800))
print(f"{d['mean_nll']} {d['top1_acc']*100}")
PY
)
set -- $LOSS
chk "mean_nll（200 token）" "${1:-99}" 2.1860 "abs(a-b)<=0.02"
chk "top1（%）"              "${2:-0}"  48.00  "abs(a-b)<=2.0"

# ---- 3. TPOT / TTFT ----
read -r TPOT TTFT <<<"$(python3 - <<'PY'
import json, time, urllib.request
URL = "http://127.0.0.1:8000/v1/chat/completions"
def gen(prompt, mx):
    body = json.dumps({"messages": [{"role": "user", "content": prompt}],
                       "max_tokens": mx, "temperature": 0, "stream": True}).encode()
    r = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); first = None; k = 0
    for line in urllib.request.urlopen(r, timeout=1800):
        s = line.decode("utf-8", "ignore").strip()
        if not s.startswith("data: "): continue
        p = s[6:]
        if p == "[DONE]": break
        d = json.loads(p)
        if d["choices"][0]["delta"].get("content"):
            k += 1
            if first is None: first = time.perf_counter() - t0
    tot = time.perf_counter() - t0
    return first, (tot - first) / max(1, k - 1) * 1e3 if k > 1 else 9999
gen("暖机", 8)
_, tpot = gen("请写一段约150字的散文，主题是秋天", 96)
f1, _ = gen("你好", 8)
_, _ = gen("你好", 8)
f2, _ = gen("你好", 8)          # 第二次及以后应命中缓存
print(f"{tpot:.1f} {f2:.2f}")
PY
)"
chk "TPOT（ms/token）" "${TPOT:-9999}" 118 "a<=135"
chk "TTFT 热态（s）"   "${TTFT:-99}"   0.14 "a<=0.5"

# ---- 4. 显存 ----
MEM=$(python3 - "$RUN" <<'PY'
import re, sys
run = sys.argv[1]
try:
    log = open(f"{run}/logs/server.log", encoding="utf-8", errors="ignore").read()
except OSError:
    print("999"); raise SystemExit
m = re.findall(r"hbm0: alloc=([0-9.]+) GiB", log)
print(m[-1] if m else "999")
PY
)
chk "dev0 显存（GiB）" "${MEM:-999}" 59.57 "a<=60.4"

# ---- 5. 24 题 MCQ（可选）----
if [ $QUICK -eq 0 ]; then
  if [ -f "$RUN/fnx/tools/mcq_eval.py" ]; then
    ACC=$(python3 "$RUN/fnx/tools/mcq_eval.py" 2>/dev/null | tail -1 | sed -n 's/.*准确率 \([0-9]*\)\/\([0-9]*\).*/\1/p')
    chk "MCQ 准确率（题数）" "${ACC:-0}" 24 "a>=22"
  else
    echo "  ..   跳过 MCQ（没找到 tools/mcq_eval.py）"
  fi
else
  echo "  ..   跳过 MCQ（--quick）"
fi

echo
if [ $fail -eq 0 ]; then
  printf '\033[32mBASELINE OK\033[0m  —— 与 baseline-r31 一致\n'
  exit 0
else
  printf '\033[31mBASELINE FAIL\033[0m —— %d 项偏离基线，检查 BASELINE.md §5 的已知坑\n' "$fail"
  exit 1
fi
