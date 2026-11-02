#!/usr/bin/env python3
"""ttft_probe.py — 流式测 TTFT（同口径，可在任意 OpenAI 兼容端点上跑）。

两种模式都测：
  cold : 每次请求带唯一随机前缀 ⇒ 前缀缓存必然不命中，TTFT 反映真实 prefill；
  warm : 同一 prompt 连发 ⇒ 前缀缓存命中，TTFT 反映"读缓存 + 首 token"的下限。

用法:
  python3 ttft_probe.py --base-url URL --model NAME [--api-key K] [--lens 300,2000] [--repeat 4] [--no-think]

口径:
  * 每个长度每个模式先跑 1 次预热（不进统计），再连跑 repeat 次；
  * TTFT = 请求发出 → 收到第一个生成 token 的墙钟时间（含 HTTP、服务端 tokenize、prefill）；
  * 首 token 认 content 或 reasoning_content（开启思考链的模型会先出 reasoning）。
"""
import argparse
import json
import statistics
import time
import urllib.request
import uuid

BASE_TEXT = ("昇腾处理器面向大规模深度学习训练与推理场景，其软件栈以 CANN 为核心。"
             "训练框架通过图模式把算子下发到 AI Core，推理侧则关注首 token 延迟与解码吞吐。")


def build_prompt(target_chars, uniq=None):
    reps = max(1, target_chars // len(BASE_TEXT))
    head = (f"[{uniq}] " if uniq else "") + "请阅读下面这段材料，并只用一句话概括它的主题：\n"
    return head + BASE_TEXT * reps


def stream_once(base_url, model, api_key, prompt, max_tokens=8, timeout=1800, no_think=False):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }
    if no_think:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(), headers=headers)
    t0 = time.perf_counter()
    first = None
    n = 0
    try:
        for line in urllib.request.urlopen(req, timeout=timeout):
            s = line.decode("utf-8", "ignore").strip()
            if not s.startswith("data: "):
                continue
            body = s[6:]
            if body == "[DONE]":
                break
            d = json.loads(body)
            ch = d.get("choices") or [{}]
            delta = ch[0].get("delta") or {}
            c = delta.get("content") or delta.get("reasoning_content")
            if c:
                n += 1
                if first is None:
                    first = time.perf_counter() - t0
                if n >= max_tokens:
                    break
    except Exception as e:  # noqa: BLE001
        return None, time.perf_counter() - t0, n, repr(e)
    return first, time.perf_counter() - t0, n, None


def run_mode(a, chars, mode):
    ttfts, errs = [], []
    fixed = build_prompt(chars)
    for i in range(a.repeat + 1):                      # 第 1 次预热
        prompt = build_prompt(chars, uniq=uuid.uuid4().hex[:8]) if mode == "cold" else fixed
        first, total, n, err = stream_once(a.base_url, a.model, a.api_key, prompt,
                                           no_think=a.no_think)
        if err or first is None:
            errs.append(err or "no content")
            continue
        if i > 0:
            ttfts.append(first)
    if not ttfts:
        print(f"  prompt≈{chars} 字符 [{mode}] 失败: {errs[:2]}")
        return
    med = statistics.median(ttfts)
    print(f"  prompt≈{chars} 字符 [{mode:4s}] TTFT 中位 {med * 1000:7.0f} ms "
          f"(min {min(ttfts) * 1000:.0f} / max {max(ttfts) * 1000:.0f}, n={len(ttfts)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default="")
    ap.add_argument("--lens", default="300,2000")
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--no-think", action="store_true",
                    help="显式关闭思考链（chat_template_kwargs.enable_thinking=false）")
    a = ap.parse_args()
    for chars in [int(x) for x in a.lens.split(",")]:
        for mode in ("cold", "warm"):
            run_mode(a, chars, mode)


if __name__ == "__main__":
    main()
