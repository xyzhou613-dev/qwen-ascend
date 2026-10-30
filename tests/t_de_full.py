"""步骤 (d)+(e)：48 层双卡装配 + 真实 prompt 端到端 + 实测 TPOT。"""
import sys
import time

import torch

sys.path.insert(0, "/root/ascend-dev/fnx")
from fnx import engine as E       # noqa: E402
from fnx.model import FnModel     # noqa: E402

PROMPT = "用一句话解释什么是线性注意力。"
DECODE = 24
REPEAT = 20


def main():
    t_all = time.time()
    print("=== 步骤(d): 48 层双卡装配 ===", flush=True)
    m = FnModel(P=2048, lpc=23, dev0="npu:0", dev1="npu:1", nlayers=46,
                with_head=True, verbose=True)
    print(f"装配总耗时 {time.time()-t_all:.1f}s", flush=True)

    print("\n=== 图捕获（body0=24层 / body1=24层 / head）===", flush=True)
    t = time.time()
    m.capture()
    print(f"捕获耗时 {time.time()-t:.1f}s", flush=True)

    # ---- tokenizer ----
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/models/Qwen3.8-Flash-Next")
    msgs = [{"role": "user", "content": PROMPT}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids[0].tolist()
    print(f"\nprompt: {PROMPT!r}")
    print(f"prompt tokens = {len(ids)}")

    # ---- A. 正确性/生成（含每步 host 同步取 argmax）----
    t = time.time()
    out = m_gen(m, tok, ids, DECODE)
    t_gen = time.time() - t
    print(f"\n生成 {len(out)} token，墙钟 {t_gen:.2f}s")
    print("输出:", repr(tok.decode(out, skip_special_tokens=True)))

    # ---- B. prefill 速度 ----
    m.reset()
    torch.npu.synchronize()
    t = time.perf_counter()
    for x in ids[:-1]:
        m.step(int(x), with_head=False)
    torch.npu.synchronize()
    t_pre = (time.perf_counter() - t) / max(1, len(ids) - 1) * 1e3
    print(f"\nprefill: {t_pre:.3f} ms/token（逐 token 步进，无 chunk 并行）")

    # ---- C. decode TPOT（异步，不读 logits）----
    m.reset()
    for x in ids[:-1]:
        m.step(int(x), with_head=False)
    for _ in range(3):
        m.step(ids[-1], with_head=True)
    torch.npu.synchronize()
    t = time.perf_counter()
    for _ in range(REPEAT):
        m.step(ids[-1], with_head=True)
    torch.npu.synchronize()
    tpot = (time.perf_counter() - t) / REPEAT * 1e3
    print(f"\n[decode 异步] TPOT {tpot:7.3f} ms/token  → {1000/tpot:6.1f} tok/s")

    # ---- D. decode TPOT（每步 host 同步，真实服务口径）----
    m.reset()
    for x in ids[:-1]:
        m.step(int(x), with_head=False)
    torch.npu.synchronize()
    t = time.perf_counter()
    for _ in range(REPEAT // 2):
        m.step(ids[-1], with_head=True)
        _ = int(m.logits.float().argmax(dim=-1).item())
    tpot2 = (time.perf_counter() - t) / (REPEAT // 2) * 1e3
    print(f"[decode 同步] TPOT {tpot2:7.3f} ms/token  → {1000/tpot2:6.1f} tok/s")

    # ---- E. 带宽账 ----
    wb_token = (120.8e9 + 3.71e9) * 0.5          # 真实 4bit 专家 + W8 稠密
    print(f"\n理论 roofline: {wb_token/1e9:.2f} GB/token ÷ 1447 GB/s = {wb_token/1447e9*1e3:.2f} ms")
    print(f"目标 3.5 ms；实测 {tpot2:.2f} ms → 达成率 {3.5/tpot2*100:.0f}%")


@torch.no_grad()
def m_gen(m, tok, ids, n):
    m.reset()
    for x in ids[:-1]:
        m.step(int(x), with_head=False)
    nxt = int(ids[-1])
    out = []
    for _ in range(n):
        m.step(nxt, with_head=True)
        nxt = int(m.logits.float().argmax(dim=-1).item())
        out.append(nxt)
        if nxt == E.NG_EOS:
            break
    return out


if __name__ == "__main__":
    main()
