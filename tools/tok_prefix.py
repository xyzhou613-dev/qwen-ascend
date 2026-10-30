import os
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(os.environ.get("FNX_SRCDIR", "/models/Qwen3.8-Flash-Next"))


def show(user):
    msgs = [{"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids[0].tolist()
    return text, ids


t1, i1 = show("你好")
t2, i2 = show("请详细解释一下深度学习里的反向传播算法是怎么工作的，包括链式法则、梯度累加、以及为什么需要激活函数的导数，请尽量详细。")
print("=== 模板全文（你好）===")
print(repr(t1))
print(f"\n总 token = {len(i1)}")
print("token 明细：")
for k, t in enumerate(i1):
    print(f"  [{k:3d}] {t:6d}  {tok.decode([t])!r}")

# 公共前缀
n = 0
for a, b in zip(i1, i2):
    if a != b:
        break
    n += 1
print(f"\n两个不同 user 内容的公共前缀 = {n} token")
print(f"  → 若做前缀缓存，'你好' 可省 {n} 步 × 127 ms = {n*0.127:.2f} s")
print(f"  → 长 prompt 可省 {n} 步 × 135 ms = {n*0.135:.2f} s")

print("\n=== 多轮对话（带历史）前缀对比 ===")
m3 = [{"role": "user", "content": "你好"}, {"role": "assistant", "content": "你好！"},
      {"role": "user", "content": "再见"}]
t3 = tok.apply_chat_template(m3, tokenize=False, add_generation_prompt=True)
i3 = tok(t3, return_tensors="pt").input_ids[0].tolist()
n3 = 0
for a, b in zip(i1, i3):
    if a != b:
        break
    n3 += 1
print(f"单轮 vs 三轮 公共前缀 = {n3} token（历史部分可缓存但每轮变化）")
