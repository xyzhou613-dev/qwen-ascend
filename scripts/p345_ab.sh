#!/bin/bash
# p345_ab.sh — A/B：MFAST=0/1 各重启一次，测 TPOT + NLL（用法: bash p345_ab.sh <0|1>）
set -u
MF=${1:-0}
echo "===== MFAST=$MF ====="
MFAST=$MF bash /root/ascend-dev/p275.sh 2>&1 | tail -7
bash /root/ascend-dev/p340_tpot.sh
python3 - <<PY
import json, urllib.request
text = open("/root/ascend-dev/eval_text.txt", encoding="utf-8").read()
rq = urllib.request.Request("http://127.0.0.1:8000/fnx/loss",
                            data=json.dumps({"text": text, "max_tokens": 200}).encode(),
                            headers={"Content-Type": "application/json"})
d = json.load(urllib.request.urlopen(rq, timeout=1800))
print("  mean_nll=%.4f  ppl=%.3f  top1=%.1f%%  n=%d" % (
    d["mean_nll"], d["ppl"], d["top1_acc"] * 100, d["n"]), flush=True)
PY
