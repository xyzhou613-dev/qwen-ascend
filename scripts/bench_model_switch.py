"""Measure modelctl switch time until the target streams its first token."""

import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request


MODELS = {"27b": "qwen3.8-27b", "flash-next": "qwen3.8-flash-next-w4"}


def request(url, body=None, timeout=5):
    data = None if body is None else json.dumps(body).encode()
    return urllib.request.urlopen(
        urllib.request.Request(url, data, {"Content-Type": "application/json"}),
        timeout=timeout,
    )


def probe(base, model, prompt, count, quality=False):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": count,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if quality:
        body.update(logprobs=True, top_logprobs=3)
    elif model == MODELS["27b"]:
        body["ignore_eos"] = True
    started = time.monotonic()
    first = last = None
    fragments, logprobs = [], []
    usage = None
    done = False
    with request(base + "/v1/chat/completions", body, timeout=90) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise ValueError(event["error"])
            for choice in event.get("choices", []):
                content = choice.get("delta", {}).get("content")
                if content:
                    last = time.monotonic()
                    if first is None:
                        first = last
                    fragments.append(content)
                logprobs.extend((choice.get("logprobs") or {}).get("content") or [])
            usage = event.get("usage") or usage
    if not done or first is None:
        raise ValueError("incomplete stream or no generated content")
    tokens = (usage or {}).get("completion_tokens")
    return {
        "first_at": first,
        "ttft_s": first - started,
        "output_tokens": tokens,
        "decode_tokens_s": (tokens - 1) / (last - first) if tokens and last > first else None,
        "text": "".join(fragments),
        "logprobs": logprobs,
    }


def wait_for_model(base, model, timeout, process):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process and process.poll() not in (None, 0):
            raise RuntimeError(f"modelctl exited with status {process.returncode}")
        try:
            with request(base + "/v1/models", timeout=1) as response:
                if model in {item["id"] for item in json.load(response)["data"]}:
                    return time.monotonic()
        except (OSError, ValueError):
            pass
        time.sleep(0.25)
    raise TimeoutError(f"{model} did not become ready")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=MODELS)
    parser.add_argument("output", type=Path)
    parser.add_argument("--modelctl", default="/root/ascend-dev/modelctl.sh")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--observe", action="store_true", help="probe the already-starting model")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = MODELS[args.target]
    with args.output.with_suffix(".switch.log").open("w") as log:
        started = time.monotonic()
        process = None if args.observe else subprocess.Popen(
            [args.modelctl, "start", args.target], stdout=log, stderr=subprocess.STDOUT
        )
        ready = wait_for_model(args.url, model, 950, process)
        warmup = probe(args.url, model, "What is 12 plus 30? Reply with just the number.", 12)
        first_at = warmup.pop("first_at")
        if process and process.wait(timeout=30):
            raise RuntimeError(f"modelctl exited with status {process.returncode}")
        result = {
            "model": model,
            "switch_to_models_s": None if args.observe else ready - started,
            "switch_to_token_s": None if args.observe else first_at - started,
            "first_request": warmup,
        }
        if args.target == "27b":
            result["decode"] = []
            for _ in range(3):
                output = probe(args.url, model, "List prime numbers in ascending order.", 64)
                output.pop("first_at")
                result["decode"].append(output)
            result["quality"] = []
            for prompt in (
                "What is 12 plus 30? Reply with just the number.",
                "Write a Python function that computes the median of a list.",
                "用中文解释什么是幂等操作，并举一个例子。",
            ):
                output = probe(args.url, model, prompt, 32, quality=True)
                output.pop("first_at")
                result["quality"].append(output)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        print({key: result[key] for key in ("model", "switch_to_models_s", "switch_to_token_s")})


if __name__ == "__main__":
    main()
