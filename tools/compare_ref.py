"""★ 配对比较：线上 W4（48 层）vs 参考 bf16（同一文本同一 token 前缀）。

用法（宿主 python3）：
  python3 tools/compare_ref.py /tmp/ref_eval12.json
参考 JSON 由 tests/t_ref_eval.py 产出（含 ids/nll/rank/mean_nll）。
我们这边用 /fnx/loss?max_tokens=12 取同样 12 个预测的逐 token NLL/rank。
"""
import json
import statistics as st
import sys
import urllib.request

CAP = 12
URL = "http://127.0.0.1:8000/fnx/loss"
EVAL = "/root/ascend-dev/eval_text.txt"


def main():
    ref = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/ref_eval12.json"))
    text = open(EVAL, encoding="utf-8").read()
    r = urllib.request.Request(URL, data=json.dumps({"text": text, "max_tokens": CAP}).encode(),
                               headers={"Content-Type": "application/json"})
    ours = json.load(urllib.request.urlopen(r, timeout=900))
    n = min(len(ref["nll"]), len(ours["nll_tokens"]))
    print(f"=== 配对比较（{n} 个预测，同一 eval_text 前缀）===")
    print(f"  参考 bf16 ({ref['nl']} 层): mean_nll = {ref['mean_nll']:.4f}")
    print(f"  线上 W4   (48 层) : mean_nll = {sum(ours['nll_tokens'][:n])/n:.4f}")
    print()
    print("   pos | 参考 bf16 NLL   rank | 我们 W4 NLL   rank | 差")
    diffs = []
    for i in range(n):
        a, b = ref["nll"][i], ours["nll_tokens"][i]
        ra = ref["rank"][i] if i < len(ref["rank"]) else -1
        rb = ours["rank_tokens"][i] if i < len(ours["rank_tokens"]) else -1
        diffs.append(b - a)
        flag = "  <<<" if abs(b - a) > 1.5 else ""
        print(f"  {i:3d} | {a:12.3f} {ra:6d} | {b:11.3f} {rb:6d} | {b-a:+7.3f}{flag}")
    print()
    print(f"  平均差（我们 − 参考）= {st.mean(diffs):+.4f} nats"
          f"   中位 {st.median(diffs):+.4f}   max {max(diffs):+.4f}   min {min(diffs):+.4f}")
    same = sum(1 for i in range(n) if ref["rank"][i] == ours["rank_tokens"][i])
    print(f"  逐 token rank 完全相同的个数 = {same}/{n}")
    d = st.mean(diffs)
    if abs(d) < 0.2:
        verdict = "量化+实现差距 < 0.2 nats（很小）→ 优先做速度/工程，不必急着换量化分组"
    elif d > 0.2:
        verdict = "差距 > 0.2 nats → 值得优先做量化（g64→g32 等）"
    else:
        verdict = "我们反而更好 → 需要复核参考实现的口径"
    print(f"  ★ 结论：{verdict}")


if __name__ == "__main__":
    main()
