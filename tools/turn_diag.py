"""多轮第 2 轮为什么命中不了前缀缓存？—— 逐 token 对比 turn1 的 prompt[:-1] 与 turn2 的 prompt。

假设（第 30 轮 §4 已发现一次）：模板在 assistant 边界插的 token 与"生成前缀"不完全一致。
本脚本用**真实回答**复现，并打印分歧点前后的 token 明细。
"""
import json
import os
import urllib.request

from transformers import AutoTokenizer

SRC = os.environ.get("FNX_SRCDIR", "/models/Qwen3.8-Flash-Next")
tok = AutoTokenizer.from_pretrained(SRC)
KW = {"enable_thinking": False}


def ids_of(msgs):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **KW)
    return [int(x) for x in tok(text, return_tensors="pt").input_ids[0].tolist()], text


def ask(msg, mx=24):
    body = json.dumps({"messages": [{"role": "user", "content": msg}], "max_tokens": mx,
                       "temperature": 0, "enable_thinking": False}).encode()
    r = urllib.request.Request("http://127.0.0.1:8000/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=600))
    return d["choices"][0]["message"]["content"], d["choices"][0]["finish_reason"]


a1, fin = ask("你好")
print(f"turn1 回答: {a1!r}  finish={fin}")

msgs1 = [{"role": "user", "content": "你好"}]
msgs2 = msgs1 + [{"role": "assistant", "content": a1},
                 {"role": "user", "content": "再说一次"}]
p1, t1 = ids_of(msgs1)
p2, t2 = ids_of(msgs2)
print(f"\nturn1 prompt {len(p1)} token / turn2 prompt {len(p2)} token")
print(f"缓存 key = turn1 prompt[:-1] = {len(p1)-1} token")

key = p1[:-1]
n = 0
for a, b in zip(key, p2):
    if a != b:
        break
    n += 1
print(f"\n★ 公共前缀 = {n} token（key 长 {len(key)}）")
lo = max(0, n - 4)
print(f"\n分歧点附近（idx {lo}..{min(len(key), len(p2), n+6)}）：")
print("  idx | key(turn1[:-1])                | turn2 prompt")
for i in range(lo, min(max(len(key), len(p2)), n + 8)):
    ka = f"{key[i]:6d} {tok.decode([key[i]])!r}" if i < len(key) else "<越界>"
    kb = f"{p2[i]:6d} {tok.decode([p2[i]])!r}" if i < len(p2) else "<越界>"
    mark = "  <<< 分歧" if i == n else ""
    print(f"  {i:3d} | {ka:32s} | {kb}{mark}")

print("\n=== turn1 模板全文 ===")
print(repr(t1))
print("\n=== turn2 模板全文 ===")
print(repr(t2))
