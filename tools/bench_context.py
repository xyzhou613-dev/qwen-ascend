"""p333_ctx_bench: 长上下文的 prefill / decode / 端到端测量（第 44 轮）。

口径（与全项目一致）：
  * TTFT      = 请求发出 → 第一个 content token，含完整 prefill + HTTP/分词
  * prefill   = ms/prefill-token，用 /fnx/prof 的 n_prefill 做分母
  * TPOT@pos  = 在 pos≈prompt_len 处生成 16 个 token 的平均间隔（首 token 不计）
  * 端到端    = 一个"读 n token prompt、写 16 token"的真实请求总耗时

用法：
  python p333_ctx_bench.py 8192 16384      # 按列出的 prompt 长度逐个测
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"
PROF = "http://127.0.0.1:8000/fnx/prof"
BASE = "昇腾处理器面向大规模深度学习训练与推理场景，其软件栈以CANN为核心。"
HEAD = "请阅读下面这段材料，并在最后用一句话概括它的主题：\n"


def build_prompt(target_chars):
    reps = max(1, target_chars // len(BASE))
    return HEAD + BASE * reps


def gen(msg, max_new=16, timeout=7200):
    body = json.dumps({"messages": [{"role": "user", "content": msg}],
                       "max_tokens": max_new, "temperature": 0,
                       "stream": True}).encode()
    req = urllib.request.Request(URL, data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    first = None
    k = 0
    for line in urllib.request.urlopen(req, timeout=timeout):
        s = line.decode("utf-8", "ignore").strip()
        if not s.startswith("data: "):
            continue
        p = s[6:]
        if p == "[DONE]":
            break
        d = json.loads(p)
        if d["choices"][0]["delta"].get("content"):
            k += 1
            if first is None:
                first = time.perf_counter() - t0
    total = time.perf_counter() - t0
    prof = json.load(urllib.request.urlopen(PROF, timeout=60))["last_prof"]
    return first, total, k, prof


def main():
    for target in [int(a) for a in sys.argv[1:]]:
        # 实测本模型中文约 0.555 token/字符（1110 token / 2000 字），按此换算；
        # target 现在表示"目标 token 数"
        msg = build_prompt(int(target * 1.8))
        print(f"... 请求 prompt≈{target} token", flush=True)
        try:
            first, total, k, prof = gen(msg)
        except Exception as e:  # noqa: BLE001
            print(f"  prompt≈{target}: 失败 {type(e).__name__}: {str(e)[:120]}", flush=True)
            continue
        n_pre = prof.get("n_prefill") or 0
        n_prompt = prof.get("n_prompt") or 0
        per = first * 1e3 / n_pre if n_pre else 0.0
        tpot = (total - (first or 0)) / max(1, k - 1) * 1e3 if k > 1 else 0.0
        print(f"  n_prompt={n_prompt:6d} n_prefill={n_pre:6d} hit={prof.get('prefix_hit',0):4d} "
              f"out={k:3d} | TTFT={first:8.2f}s ({per:6.1f} ms/prefill-tok) "
              f"TPOT@pos≈{n_prompt}={tpot:6.1f}ms | 端到端={total:8.2f}s", flush=True)


if __name__ == "__main__":
    main()
