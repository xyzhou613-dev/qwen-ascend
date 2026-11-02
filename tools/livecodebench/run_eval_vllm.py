#!/usr/bin/env python3
"""LiveCodeBench code_generation_lite 100 题编码能力评测.

流程: 读配置 -> 构造官方 prompt -> 调 OpenAI 兼容接口生成 -> 抽取代码块
      -> 子进程判题 -> 输出 pass@1 通过率 + 吞吐量.

用法:
    python3 scripts/run_eval.py                     # 使用 config.json
    python3 scripts/run_eval.py --model X --base-url http://h:p/v1
    python3 scripts/run_eval.py --skip-eval         # 只生成不判题
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from testing_util import grade_generation  # noqa: E402

SYSTEM_MESSAGE = (
    "You are an expert Python programmer. You will be given a question "
    "(problem specification) and will generate a correct Python program "
    "that matches the specification and passes all tests."
)
FMT_STARTER = (
    "You will use the following starter code to write the solution to the "
    "problem and enclose your code within delimiters."
)
FMT_STDIN = (
    "Read the inputs from stdin solve the problem and write the answer to "
    "stdout (do not directly test on the sample inputs). Enclose your code "
    "within delimiters as follows. Ensure that when the python program "
    "runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)


FMT_FUNC = (
    "Write your solution as a Python function named `{fn}` (keep this exact name) and "
    "enclose your code within delimiters. Read nothing from stdin; the tests import your "
    "code and call `{fn}(...)` directly with the arguments described in the question."
)


def build_messages(problem):
    p = f"### Question:\n{problem['question_content']}\n\n"
    fn = problem.get("func_name")
    if problem.get("starter_code"):
        p += f"### Format: {FMT_STARTER}\n```python\n{problem['starter_code']}\n```\n\n"
    elif fn:
        # ★ 修正：MBPP+ 这类 functional 题没有 starter_code，原版会退化成"stdin 模式"提示词，
        #   而判题器是按 testtype=script 直接调用入口函数名的 → 必然 NameError。
        #   这里显式给出入口函数名，保持与原版一致的其余措辞。
        p += (f"### Format: {FMT_FUNC.format(fn=fn)}"
              f"\n```python\n# YOUR CODE HERE\n```\n\n")
    else:
        p += f"### Format: {FMT_STDIN}\n```python\n# YOUR CODE HERE\n```\n\n"
    p += "### Answer: (use the provided format with backticks)\n\n"
    return [{"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": p}]



def extract_code(text):
    """官方 extract_code: 去掉 <think> 块, 取最后一个 ``` 代码块."""
    if not text:
        return ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    lines = text.split("\n")
    idx = [i for i, l in enumerate(lines) if "```" in l]
    if len(idx) < 2:
        # 无围栏兜底: 从第一行像代码的行起取
        starts = ("import ", "from ", "def ", "class ", "#!/", "if __name__")
        s = next((i for i, l in enumerate(lines) if l.startswith(starts)), None)
        return "\n".join(lines[s:]).strip() if s is not None else ""
    return "\n".join(lines[idx[-2] + 1:idx[-1]]).strip()


def call_model(cfg, problem):
    payload = {
        "model": cfg["model"],
        "messages": build_messages(problem),
        "temperature": cfg.get("temperature", 0.0),
        "top_p": cfg.get("top_p", 1.0),
        "max_tokens": cfg.get("max_tokens", 8192),
        "n": cfg.get("n", 1),
        "stream": False,
    }
    # ★ 只加这一处：把 config 里的 chat_template_kwargs 传下去（用于关掉 27B 的默认思考链，
    #   以便与 Flash-Next（FNX_ENABLE_THINKING=0）的口径一致）。不设则无影响。
    if cfg.get("chat_template_kwargs"):
        payload["chat_template_kwargs"] = cfg["chat_template_kwargs"]

    headers = {"Authorization": f"Bearer {cfg.get('api_key', 'EMPTY')}"}
    t0 = time.time()
    try:
        r = requests.post(f"{cfg['base_url'].rstrip('/')}/chat/completions",
                          json=payload, headers=headers,
                          timeout=cfg.get("request_timeout", 900))
        dt = time.time() - t0
        r.raise_for_status()
        body = r.json()
    except Exception as e:
        return {"error": repr(e), "latency_s": time.time() - t0,
                "gens": [], "prompt_tokens": 0, "completion_tokens": 0}
    usage = body.get("usage") or {}
    gens = [c.get("message", {}).get("content", "") or ""
            for c in body.get("choices", [])]
    return {"gens": gens, "latency_s": dt,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "error": None}


def _grade_one(job):
    pid, gi, code, timeout, problem = job
    passed, detail = grade_generation(problem, code, timeout)
    return pid, gi, passed, detail


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.json"))
    ap.add_argument("--base-url")
    ap.add_argument("--model")
    ap.add_argument("--api-key")
    ap.add_argument("--dataset", default=str(ROOT / "dataset/lcb_sample100.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    cfg = json.loads(open(args.config).read())
    for k, v in [("base_url", args.base_url), ("model", args.model),
                 ("api_key", args.api_key)]:
        if v:
            cfg[k] = v

    problems = [json.loads(l) for l in open(args.dataset, encoding="utf-8")]
    n = cfg.get("n", 1)
    print(f"endpoint : {cfg['base_url']}  model={cfg['model']}  "
          f"n={n} temp={cfg.get('temperature')} problems={len(problems)}")

    # ---------- 生成阶段 ----------
    t_gen0 = time.time()
    with ThreadPoolExecutor(max_workers=cfg.get("concurrency", 4)) as ex:
        gen_out = list(ex.map(lambda p: call_model(cfg, p), problems))
    gen_wall = time.time() - t_gen0

    for i, (p, g) in enumerate(zip(problems, gen_out)):
        if g["error"]:
            print(f"[{i}] {p['platform']}/{p['difficulty']} "
                  f"{p['question_id']}  REQUEST FAILED: {g['error']}")

    codes = [[extract_code(o) for o in g["gens"]] for g in gen_out]
    tot_ctok = sum(g["completion_tokens"] for g in gen_out)
    tot_ptok = sum(g["prompt_tokens"] for g in gen_out)

    # ---------- 判题阶段 ----------
    per_problem = []
    if not args.skip_eval:
        jobs = [(p["idx"], gi, c, cfg.get("test_timeout", 20), p)
                for p, cs in zip(problems, codes) for gi, c in enumerate(cs)]
        t_eval0 = time.time()
        graded = {}
        with ProcessPoolExecutor(max_workers=cfg.get("eval_workers", 8)) as ex:
            for pid, gi, passed, detail in ex.map(_grade_one, jobs):
                graded.setdefault(pid, []).append((gi, passed, detail))
        eval_wall = time.time() - t_eval0
        print(f"eval wall time: {eval_wall:.1f}s")
    else:
        graded = {}
        eval_wall = 0.0

    # ---------- 汇总 ----------
    n_pass = 0
    print("\n" + "=" * 88)
    print(f"{'#':>2} {'platform':<10} {'difficulty':<7} {'qid':<12} "
          f"{'tests':>5} {'pass':>5} {'ctok':>6} {'lat(s)':>7} {'tok/s':>6}")
    print("-" * 88)
    for i, (p, g) in enumerate(zip(problems, gen_out)):
        pid = p["idx"]
        if args.skip_eval or pid not in graded:
            p1, ntest = None, len(p["tests"])
        else:
            gens_res = sorted(graded[pid])
            p1 = sum(x[1] for x in gens_res) / max(len(gens_res), 1)
            ntest = len(p["tests"])
            if p1 == 1.0:
                n_pass += 1
        tps = g["completion_tokens"] / g["latency_s"] if g["latency_s"] else 0
        ps = f"{p1 * 100:4.0f}%" if p1 is not None else "  - "
        print(f"{i:>2} {p['platform']:<10} {p['difficulty']:<7} "
              f"{p['question_id']:<12} {ntest:>5} {ps:>5} "
              f"{g['completion_tokens']:>6} {g['latency_s']:>7.1f} {tps:>6.1f}")
    print("=" * 88)

    pass_rate = n_pass / len(problems) if not args.skip_eval else None
    agg_tps = tot_ctok / gen_wall if gen_wall else 0
    print(f"\npass@1 (problem-level): "
          f"{'n/a' if pass_rate is None else f'{n_pass}/{len(problems)} = {pass_rate * 100:.1f}%'}")
    print(f"generation wall time  : {gen_wall:.1f}s")
    print(f"prompt tokens         : {tot_ptok}")
    print(f"completion tokens     : {tot_ctok}")
    print(f"throughput (tok/s)    : {agg_tps:.1f}  "
          f"(agg), {tot_ctok / sum(g['latency_s'] for g in gen_out):.1f} (sum-latency basis)")

    # ---------- 落盘 ----------
    outdir = Path(args.out) if args.out else (
        ROOT / "results" /
        f"{cfg['model'].replace('/', '_')}_{datetime.now():%Y%m%d_%H%M%S}")
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "generations.jsonl", "w", encoding="utf-8") as f:
        for p, g, cs in zip(problems, gen_out, codes):
            f.write(json.dumps({
                "idx": p["idx"], "question_id": p["question_id"],
                "platform": p["platform"], "difficulty": p["difficulty"],
                "latency_s": g["latency_s"],
                "prompt_tokens": g["prompt_tokens"],
                "completion_tokens": g["completion_tokens"],
                "error": g["error"],
                "raw_outputs": g["gens"], "codes": cs,
            }, ensure_ascii=False) + "\n")
    summary = {
        "model": cfg["model"], "base_url": cfg["base_url"],
        "n_samples_per_problem": n,
        "dataset": str(args.dataset),
        "pass_at_1": pass_rate,
        "problems_passed": n_pass if not args.skip_eval else None,
        "problems_total": len(problems),
        "gen_wall_time_s": gen_wall,
        "eval_wall_time_s": eval_wall,
        "total_prompt_tokens": tot_ptok,
        "total_completion_tokens": tot_ctok,
        "throughput_completion_tok_per_s": agg_tps,
        "per_problem": [
            {"idx": p["idx"], "question_id": p["question_id"],
             "platform": p["platform"], "difficulty": p["difficulty"],
             "n_tests": len(p["tests"]),
             "pass": (None if args.skip_eval or p["idx"] not in graded
                      else sum(x[1] for x in graded[p["idx"]]) / len(graded[p["idx"]])),
             "latency_s": g["latency_s"],
             "prompt_tokens": g["prompt_tokens"],
             "completion_tokens": g["completion_tokens"],
             "error": g["error"]}
            for p, g in zip(problems, gen_out)],
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nresults saved to {outdir}")


if __name__ == "__main__":
    main()
