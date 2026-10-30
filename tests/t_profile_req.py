"""★ 定位每请求 ~5.7 s 的固定开销：复刻 server.generate() 的循环，逐段计时。

怀疑点（全部单独计时）：
  1 reset()                —— 清零 S/conv/KV + torch.npu.synchronize()
  2 每个 prefill step      —— 逐个 token，with_head=False
  3 每个 decode step       —— with_head=True
  4 采样那一下             —— logits.argmax().item()（D2H 同步）
"""
import os
import time

import torch

from fnx.model import FnModel

LPC = int(os.environ.get("PR_LPC", "44"))
NL = int(os.environ.get("PR_NL", "46"))
MAXNEW = int(os.environ.get("PR_MAXNEW", "8"))


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get("FNX_SRCDIR",
                                                       "/models/Qwen3.8-Flash-Next"))
    msgs = [{"role": "user", "content": "你好"}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt").input_ids[0].tolist()
    print(f"=== 复刻 generate(): prompt={len(ids)} token, max_new={MAXNEW} ===", flush=True)

    m = FnModel(P=2048, lpc=LPC, dev0="npu:0", dev1="npu:1", nlayers=NL,
                with_head=True, with_ngram=False, verbose=False)
    m.capture()
    print(f"  dev0={len(m.layers0)} 层 dev1={len(m.layers1)} 层  g0={'有' if m.g0 else '无'}",
          flush=True)

    T = lambda: time.perf_counter()
    with torch.no_grad():
        # 预热一次完整请求
        m.reset()
        for x in ids[:-1]:
            m.step(int(x), with_head=False)
        m.step(int(ids[-1]), with_head=True)
        torch.npu.synchronize()

        # ---- 正式计时 ----
        t0 = T()
        m.reset()
        torch.npu.synchronize()
        t_reset = T() - t0

        pre = []
        for x in ids[:-1]:
            a = T()
            m.step(int(x), with_head=False)
            torch.npu.synchronize()
            pre.append(T() - a)

        dec, samp = [], []
        nxt = int(ids[-1])
        for _ in range(MAXNEW):
            a = T()
            m.step(nxt, with_head=True)
            torch.npu.synchronize()
            dec.append(T() - a)
            b = T()
            lg = m.logits.float().view(-1)
            nxt = int(lg.argmax().item())
            samp.append(T() - b)

    print(f"\n  reset()            {t_reset*1e3:9.1f} ms", flush=True)
    print(f"  prefill 共 {len(pre)} 步  {sum(pre)*1e3:9.1f} ms  "
          f"（首步 {pre[0]*1e3:.1f} / 其余均值 {sum(pre[1:])/max(1,len(pre)-1)*1e3:.1f} ms）",
          flush=True)
    print(f"  decode  共 {len(dec)} 步  {sum(dec)*1e3:9.1f} ms  "
          f"（均值 {sum(dec)/len(dec)*1e3:.1f} ms）", flush=True)
    print(f"  采样合计           {sum(samp)*1e3:9.1f} ms（均值 {sum(samp)/len(samp)*1e3:.1f}）",
          flush=True)
    total = t_reset + sum(pre) + sum(dec) + sum(samp)
    print(f"\n  ★ 模型侧合计 {total*1e3/1e3:.2f} s", flush=True)
    print(f"  ★ 对照：实测整请求 6.74 s → 差值 {(6.74-total):+.2f} s 属于 HTTP/分词/框架开销",
          flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
