"""为什么输出里 'MoE' 变成 'Mo'、'Qwen' 变成 'Q'？先排除分词器/解码器的问题。"""
import os

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(os.environ.get("FNX_SRCDIR", "/models/Qwen3.8-Flash-Next"))

for s in ["MoE", "Qwen", "Mo", "Q", "MoE（混合专家）", "通义千问（Qwen）"]:
    ids = tok(s, add_special_tokens=False).input_ids
    back = tok.decode(ids, skip_special_tokens=False)
    back_skip = tok.decode(ids, skip_special_tokens=True)
    print(f"{s!r:22s} ids={ids} decode={back!r} skip={back_skip!r} 一致={back == s}")

print("\n--- 单独看这几个 token ---")
for s in ["E", "E）", "wen", "wen）"]:
    ids = tok(s, add_special_tokens=False).input_ids
    print(f"  {s!r:6s} ids={ids} decode={[tok.decode([i], skip_special_tokens=False) for i in ids]}")

print("\n--- 空/特殊 token 扫描（前 300 个是否存在解码为空）---")
weird = []
for i in range(min(300, len(tok))):
    d = tok.decode([i], skip_special_tokens=False)
    if d.strip() == "" and not d.isspace():
        weird.append((i, repr(d)))
print("  ", weird[:20] if weird else "无")
print("vocab_size =", len(tok))
