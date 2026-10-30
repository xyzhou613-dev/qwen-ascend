"""模板里的 reasoning effort 能不能调小 / 关掉？

现状：每个请求的静态前缀固定 45 token，其中 38 个是 system 里那段
"Reasoning effort is set to xhigh. Please think carefully..." —— 值 5.7 s/请求。
如果模板支持把 effort 调低或关掉，前缀变短、思考也变短，是双赢。
"""
import os
import inspect

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(os.environ.get("FNX_SRCDIR", "/models/Qwen3.8-Flash-Next"))
tpl = tok.chat_template or ""
if isinstance(tpl, dict):
    tpl = tpl.get("default", "")
print("=== 模板里与 effort/thinking 相关的片段 ===")
for i, line in enumerate(tpl.splitlines()):
    if any(k in line.lower() for k in ("effort", "thinking", "think_", "reason")):
        print(f"  {i:4d}| {line.strip()[:160]}")
print(f"\n模板总行数 = {len(tpl.splitlines())}")

print("\n=== 试不同 kwargs 的前缀长度 ===")
SENT = "\u0002FNXPFX\u0002"
for kw in ({}, {"reasoning_effort": "low"}, {"reasoning_effort": "medium"},
           {"reasoning_effort": "minimal"}, {"enable_thinking": False},
           {"thinking": False}, {"reasoning_effort": "none"}):
    try:
        text = tok.apply_chat_template([{"role": "user", "content": SENT}],
                                       tokenize=False, add_generation_prompt=True, **kw)
        cut = text.find(SENT)
        n = len(tok(text[:cut], return_tensors="pt").input_ids[0]) if cut > 0 else -1
        head = text[:70].replace("\n", "\\n")
        print(f"  {str(kw):34s} 前缀={n:3d} token  head={head!r}")
    except Exception as ex:
        print(f"  {str(kw):34s} 失败: {type(ex).__name__}: {str(ex)[:90]}")
