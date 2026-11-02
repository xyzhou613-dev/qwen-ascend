#!/usr/bin/env python3
"""汇总 /models/eval/results/run_{1,2,3} 的三轮结果：总体 pass@1、分平台/难度、
截断统计、吞吐、跨轮一致性，输出一段可直接粘进报告的 Markdown 表格。"""
import json
import os
import sys
from collections import defaultdict

BASE = "/models/eval/results"


def load(tag):
    p = os.path.join(BASE, tag, "summary.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p, encoding="utf-8"))
    # 补 generations.jsonl 里的代码长度等信息（可选）
    return d


def pct(v):
    return "-" if v is None else f"{v * 100:.1f}%"


def main():
    tags = ["run_1", "run_2", "run_3"]
    runs = {t: load(t) for t in tags}
    done = [t for t in tags if runs[t]]
    if not done:
        print("尚无完成的结果")
        return

    print("### 3.1 总览\n")
    hdr = "| 轮次 | 通过 | pass@1 | 生成时长 | 判题时长 | 完成 token | 吞吐 (tok/s) |"
    print(hdr)
    print("|---|---:|---:|---:|---:|---:|---:|")
    for t in done:
        d = runs[t]
        print(f"| {t} | {d['problems_passed']}/{d['problems_total']} | "
              f"{pct(d['pass_at_1'])} | {d['gen_wall_time_s'] / 60:.1f} min | "
              f"{d['eval_wall_time_s'] / 60:.1f} min | {d['total_completion_tokens']} | "
              f"{d['throughput_completion_tok_per_s']:.1f} |")
    rates = [runs[t]["pass_at_1"] for t in done]
    if len(rates) > 1:
        print(f"\n三轮 pass@1：{[pct(r) for r in rates]}；"
              f"极差 {max(rates) - min(rates):+.1%}，均值 {sum(rates) / len(rates):.1%}")
    print()

    print("### 3.2 分平台 / 分难度（各轮）\n")
    for t in done:
        d = runs[t]
        agg = defaultdict(lambda: [0, 0])
        for p in d["per_problem"]:
            key = p["platform"]
            agg[key][1] += 1
            if p["pass"] == 1.0:
                agg[key][0] += 1
        bydiff = defaultdict(lambda: [0, 0])
        for p in d["per_problem"]:
            bydiff[p["difficulty"]][1] += 1
            if p["pass"] == 1.0:
                bydiff[p["difficulty"]][0] += 1
        print(f"**{t}**")
        print("| 平台 | 通过/总 | | 难度 | 通过/总 |")
        print("|---|---:|---|---|---:|")
        plats = sorted(agg)
        diffs = ["easy", "medium", "hard"]
        for i in range(max(len(plats), len(diffs))):
            pl = plats[i] if i < len(plats) else ""
            c = f"{agg[pl][0]}/{agg[pl][1]}" if pl else ""
            df = diffs[i] if i < len(diffs) else ""
            cd = f"{bydiff[df][0]}/{bydiff[df][1]}" if df else ""
            print(f"| {pl} | {c} | | {df} | {cd} |")
        print()

    print("### 3.3 截断与耗时\n")
    print("| 轮次 | 命中 2048 上限的题数 | 生成 token 中位数 | 单题时延中位数 | 最慢一题 |")
    print("|---|---:|---:|---:|---:|")
    for t in done:
        d = runs[t]
        pp = d["per_problem"]
        trunc = sum(1 for p in pp if p["completion_tokens"] >= 2048)
        ct = sorted(p["completion_tokens"] for p in pp)
        lat = sorted(p["latency_s"] for p in pp)
        med = ct[len(ct) // 2]
        medl = lat[len(lat) // 2]
        print(f"| {t} | {trunc} | {med} | {medl:.1f} s | {max(lat):.1f} s |")
    print()

    if len(done) > 1:
        print("### 3.4 跨轮一致性\n")
        base = {p["idx"]: p["pass"] for p in runs[done[0]]["per_problem"]}
        for t in done[1:]:
            cur = {p["idx"]: p["pass"] for p in runs[t]["per_problem"]}
            same = sum(1 for k in base if base[k] == cur.get(k))
            flips = [k for k in base if base[k] != cur.get(k)]
            print(f"{done[0]} vs {t}: 逐题通过情况一致 {same}/{len(base)}"
                  + (f"，不一致题号 {flips}" if flips else "（完全一致）"))
        print()

    # 失败题清单（逐轮都失败的题，便于人工抽查）
    print("### 3.5 三轮都未通过的题\n")
    if len(done) == len(tags):
        allpass = defaultdict(list)
        for t in done:
            for p in runs[t]["per_problem"]:
                allpass[p["idx"]].append(p["pass"])
        always_fail = [k for k, v in allpass.items() if all(x == 0 for x in v)]
        info = {p["idx"]: p for p in runs[done[0]]["per_problem"]}
        print(f"共 {len(always_fail)} 题")
        print("\n| idx | qid | 平台 | 难度 | 测试数 |")
        print("|---:|---|---|---|---:|")
        for k in sorted(always_fail):
            p = info[k]
            print(f"| {k} | {p['question_id']} | {p['platform']} | {p['difficulty']} | {p['n_tests']} |")


if __name__ == "__main__":
    main()
