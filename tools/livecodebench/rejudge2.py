#!/usr/bin/env python3
"""rejudge2.py — 用修正后的判题器对已生成的 generations.jsonl 重新判题（串行、进程内）。

为什么串行：ProcessPoolExecutor 的 fork 子进程里再 Popen(preexec_fn=_limits) 会假失败
（同一题在进程内判是 True，经进程池就变 runtime），所以这里不做并行。

用法: python3 rejudge2.py <run_dir> <dataset.jsonl> [timeout]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from testing_util import grade_generation  # noqa: E402


def main():
    rundir = Path(sys.argv[1])
    dataset = Path(sys.argv[2])
    timeout = int(sys.argv[3]) if len(sys.argv) > 3 else 20

    probs = [json.loads(l) for l in open(dataset, encoding="utf-8")]
    by_id = {p["question_id"]: p for p in probs}
    gens = [json.loads(l) for l in open(rundir / "generations.jsonl", encoding="utf-8")]

    per, n_pass = [], 0
    byp = defaultdict(lambda: [0, 0])
    for g in gens:
        p = by_id[g["question_id"]]
        res = [grade_generation(p, code, timeout)[0] for code in g["codes"]]
        p1 = sum(1 for x in res if x) / max(1, len(res))
        if p1 == 1.0:
            n_pass += 1
        byp[g["platform"]][1] += 1
        if p1 == 1.0:
            byp[g["platform"]][0] += 1
        per.append({"idx": g["idx"], "question_id": g["question_id"],
                    "platform": g["platform"], "difficulty": g["difficulty"],
                    "n_tests": len(p["tests"]), "pass": p1,
                    "completion_tokens": g["completion_tokens"],
                    "latency_s": g["latency_s"]})

    print(str(rundir))
    print("  pass@1 = %d/%d = %.1f%%" % (n_pass, len(gens), n_pass / len(gens) * 100))
    print("  by platform:", {k: "%d/%d" % tuple(v) for k, v in sorted(byp.items())})
    fails = [(p["question_id"], p["platform"]) for p in per if p["pass"] != 1.0]
    print("  fails (%d):" % len(fails), fails)
    with open(rundir / "summary_rejudged.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": str(dataset), "problems_total": len(gens),
                   "problems_passed": n_pass, "pass_at_1": n_pass / len(gens),
                   "judge": "testing_util.grade_script (sequential)",
                   "per_problem": per}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
