"""★ 前缀缓存等价性 + 收益实测（必须独占 NPU，先停 server）。

口径：同一 prompt 两次跑，A=reset+全量 prefill，B=load_state(前缀快照)+剩余 prefill，
两次的首个 logits 必须逐位相同（maxdiff 应为 0.000e+00），否则前缀缓存不可上线。
"""
import os
import time

import torch

from fnx.model import FnModel

LPC = int(os.environ.get("PR_LPC", "44"))
NL = int(os.environ.get("PR_NL", "46"))
SENT = "\u0002FNXPFX\u0002"


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get("FNX_SRCDIR",
                                                       "/models/Qwen3.8-Flash-Next"))
    m = FnModel(P=2048, lpc=LPC, dev0="npu:0", dev1="npu:1", nlayers=NL,
                with_head=True, with_ngram=False, verbose=False)
    m.capture()

    t = time.time()
    text = tok.apply_chat_template([{"role": "user", "content": SENT}],
                                   tokenize=False, add_generation_prompt=True)
    pids = tok(text[:text.find(SENT)], return_tensors="pt").input_ids[0].tolist()
    m.reset()
    for x in pids:
        m.step(int(x), with_head=False)
    torch.npu.synchronize()
    t_build = time.time() - t
    state = m.save_state()
    mb = m.state_bytes(state) / 2 ** 20
    print(f"前缀 = {len(pids)} token, 建缓存 {t_build:.2f}s, 状态 {mb:.1f} MiB", flush=True)

    for p in ["你好", "请详细解释一下深度学习里的反向传播算法是怎么工作的。", "1+1等于几？"]:
        txt = tok.apply_chat_template([{"role": "user", "content": p}],
                                      tokenize=False, add_generation_prompt=True)
        ids = tok(txt, return_tensors="pt").input_ids[0].tolist()
        ok = ids[:len(pids)] == pids
        a = time.perf_counter()
        m.reset()
        for x in ids[:-1]:
            m.step(int(x), with_head=False)
        m.step(int(ids[-1]))
        torch.npu.synchronize()
        ta = time.perf_counter() - a
        la = m.logits.float().clone()
        b = time.perf_counter()
        m.load_state(state)
        for x in ids[len(pids):-1]:
            m.step(int(x), with_head=False)
        m.step(int(ids[-1]))
        torch.npu.synchronize()
        tb = time.perf_counter() - b
        lb = m.logits.float().clone()
        d = float((la - lb).abs().max())
        i1, i2 = int(la.argmax()), int(lb.argmax())
        print(f"  {p[:14]!r:18s} n={len(ids):3d} 前缀对齐={ok} "
              f"剩余prefill={len(ids)-1-len(pids):2d} "
              f"maxdiff={d:.3e} argmax {i1}/{i2} {'OK' if d == 0.0 else '**不一致**'} "
              f"耗时 {ta*1e3:.0f} -> {tb*1e3:.0f} ms", flush=True)

    # 连发 5 次同一请求，看稳定收益
    txt = tok.apply_chat_template([{"role": "user", "content": "你好"}],
                                 tokenize=False, add_generation_prompt=True)
    ids = tok(txt, return_tensors="pt").input_ids[0].tolist()
    t = time.perf_counter()
    for _ in range(5):
        m.load_state(state)
        for x in ids[len(pids):-1]:
            m.step(int(x), with_head=False)
        for _ in range(8):
            m.step(int(ids[-1]))
    torch.npu.synchronize()
    print(f"\n5 次「你好」+8 新 token（前缀缓存）= {(time.perf_counter()-t)/5*1e3:.0f} ms/次",
          flush=True)
    t = time.perf_counter()
    for _ in range(2):
        m.reset()
        for x in ids[:-1]:
            m.step(int(x), with_head=False)
        for _ in range(8):
            m.step(int(ids[-1]))
    torch.npu.synchronize()
    print(f"2 次「你好」+8 新 token（全量）  = {(time.perf_counter()-t)/2*1e3:.0f} ms/次",
          flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
