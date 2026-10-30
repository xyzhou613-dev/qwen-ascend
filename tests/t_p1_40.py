"""P1 收尾：40 轮对话（20 prompt x 2 遍，第二遍倒序）+ 一致性与漂移检查。

检查项
  1. 40/40 请求 200，无异常；输出非空
  2. 跨请求污染：同一 prompt 在两遍中输出**逐位相同**（temperature=0）
  3. 性能漂移：每轮 tpot 不随轮次恶化（前 5 轮 vs 后 5 轮）
  4. 退化/NaN 迹象：每轮 distinct token 数 ≥ 2（全同 token 视为可疑）
  5. 流式接口：每 5 轮取 1 次 stream，记录 ttft/tpot 服务口径

用法（容器内）：
  /root/ascend-dev/venv/bin/python tests/t_p1_40.py --url http://127.0.0.1:8000 --max-new 12
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROMPTS = [
    "你好",
    "什么是线性注意力？",
    "用一句话解释 MoE。",
    "1+1 等于几？",
    "写一个 Python 函数求斐波那契数列第 n 项。",
    "Explain what a tensor core does in one sentence.",
    "列举三种常见的量化误差度量方式。",
    "把这句话翻译成英文：昇腾 910B4 的显存是 64GB。",
    "什么是分组量化（group-wise quantization）？",
    "请说明 KV cache 在推理加速中的作用。",
    "为什么 4-bit 权重量化需要校准？",
    "总结一下稀疏注意力的优缺点。",
    "写一段 200 字左右的文字，介绍注意力机制的演进。",
    "给出一个 int8 反量化的数学表达式。",
    "在昇腾 NPU 上做算子开发需要注意什么？",
    "如果模型输出是乱码，可能有哪些原因？",
    "解释 roofline 模型，并说明它在推理优化中的用途。",
    "什么是投机解码（speculative decoding）？",
    "请给出一个 5 行的快速排序实现。",
    "最后一个问题：请用一句话总结这次对话。",
]


def post(url, body, timeout=600):
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def post_stream(url, body, timeout=600):
    """解析 SSE，返回 (text, ttft_ms, tpot_ms, n_tokens)。"""
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    parts = []
    stats = {}
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            ch = d.get("choices") or [{}]
            piece = (ch[0].get("delta") or {}).get("content")
            if piece:
                if first is None:
                    first = time.perf_counter() - t0
                parts.append(piece)
                n += 1
            if "fnx_stats" in d:
                stats = d["fnx_stats"]
    return "".join(parts), (first or 0.0) * 1e3, stats, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--max-new", type=int, default=12)
    ap.add_argument("--out", default="/root/ascend-dev/logs/p1_40round.jsonl")
    args = ap.parse_args()

    base = args.url.rstrip("/")
    print(f"[p1] start {time.strftime('%F %T')} url={base} max_new={args.max_new}", flush=True)
    h = post(base + "/health", {}) if False else json.loads(
        urllib.request.urlopen(base + "/health", timeout=30).read().decode())
    print(f"[p1] health {h}", flush=True)

    order = list(range(len(PROMPTS))) + list(reversed(range(len(PROMPTS))))
    recs = []
    fails = 0
    fout = open(args.out, "w", encoding="utf-8")

    for i, pi in enumerate(order):
        prompt = PROMPTS[pi]
        body = {"model": "qwen3.8-flash-next-w4", "messages": [{"role": "user", "content": prompt}],
                "max_tokens": args.max_new, "temperature": 0.0, "stream": False}
        t0 = time.perf_counter()
        try:
            r = post(base + "/v1/chat/completions", body)
            dt = time.perf_counter() - t0
            txt = r["choices"][0]["message"]["content"]
            usage = r["usage"]
            st = r.get("fnx_stats", {})
            err = None
        except urllib.error.HTTPError as e:
            dt = time.perf_counter() - t0
            txt, usage, st = "", {}, {}
            err = f"HTTP {e.code}: {e.read()[:300]!r}"
        except Exception as e:  # noqa: BLE001
            dt = time.perf_counter() - t0
            txt, usage, st = "", {}, {}
            err = f"{type(e).__name__}: {e}"
        if err:
            fails += 1
        hsh = hashlib.sha1(txt.encode("utf-8")).hexdigest()[:12]
        rec = {"rnd": i + 1, "pi": pi, "pass": 1 if i < len(PROMPTS) else 2,
               "sec": round(dt, 3), "ptok": usage.get("prompt_tokens"),
               "ntok": usage.get("completion_tokens"),
               "tpot": st.get("tpot_ms"), "sha": hsh, "len": len(txt),
               "txt": txt[:120], "err": err}
        recs.append(rec)
        fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fout.flush()
        print(f"[p1] {i+1:02d}/40 pass{rec['pass']} p{pi:02d} sec={dt:6.2f} "
              f"ptok={rec['ptok']} ntok={rec['ntok']} tpot={rec['tpot']} sha={hsh}"
              + (f" ERR={err}" if err else ""), flush=True)
        if dt > 120:
            print("[p1] WARN 单轮 >120s，后续 prompt 可能过长", flush=True)

    # 流式抽样
    print("[p1] === stream 抽样 ===", flush=True)
    streams = []
    for pi in (0, 4, 9, 14, 19):
        body = {"model": "qwen3.8-flash-next-w4",
                "messages": [{"role": "user", "content": PROMPTS[pi]}],
                "max_tokens": 8, "temperature": 0.0, "stream": True}
        try:
            txt, ttft, st, n = post_stream(base + "/v1/chat/completions", body)
            print(f"[p1] stream p{pi:02d} ttft={ttft:.1f}ms tpot={st.get('tpot_ms')} "
                  f"n={n} len={len(txt)}", flush=True)
            streams.append({"pi": pi, "ttft": ttft, "st": st, "n": n, "txt": txt})
        except Exception as e:  # noqa: BLE001
            print(f"[p1] stream p{pi:02d} FAIL {type(e).__name__}: {e}", flush=True)
            streams.append({"pi": pi, "err": str(e)})

    # 一致性
    by_pi = {}
    for r in recs:
        by_pi.setdefault(r["pi"], []).append(r)
    bad = [pi for pi, rs in by_pi.items() if len({x["sha"] for x in rs}) > 1]
    print(f"\n[p1] === 汇总 ===", flush=True)
    print(f"[p1] 请求成功 {len(recs)-fails}/{len(recs)}  失败 {fails}", flush=True)
    print(f"[p1] 跨请求污染：{len(bad)} 个 prompt 两遍输出不一致 "
          f"{'PASS' if not bad else 'FAIL ' + str(bad)}", flush=True)
    deg = []
    for r in recs:
        t = (r["txt"] or "").strip()
        if r["err"] is None and (len(t) == 0 or len(set(t)) < 2):
            deg.append(r["rnd"])
    print(f"[p1] 退化检查（空或单一字符）：{len(deg)} 轮 {deg if deg else ''} "
          f"{'PASS' if not deg else 'WARN'}", flush=True)
    tp = [r["tpot"] for r in recs if r["tpot"]]
    if tp:
        head = sum(tp[:5]) / max(1, len(tp[:5]))
        tail = sum(tp[-5:]) / max(1, len(tp[-5:]))
        print(f"[p1] TPOT 前5均 {head:.1f}ms / 后5均 {tail:.1f}ms / "
              f"全体 {sum(tp)/len(tp):.1f}ms / min {min(tp):.1f} / max {max(tp):.1f}", flush=True)
        print(f"[p1] 漂移 {100*(tail-head)/head:+.1f}% "
              f"{'PASS' if abs(tail-head)/head < 0.15 else 'WARN'}", flush=True)
    print(f"[p1] end {time.strftime('%F %T')}", flush=True)
    fout.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
