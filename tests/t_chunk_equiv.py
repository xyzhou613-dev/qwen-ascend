"""t_chunk_equiv: 分块 prefill 的数值等价性与速度（第 46 轮）。

判定标准（沿用项目惯例）：
  * logits maxdiff ≤ 1e-3（相对每行 |logits| 的均值）
  * 同一 prompt 的 teacher-forcing NLL 差 ≤ 0.05 nats
  * 再比 prefill 墙钟

用法（容器内）：
  FNX_CHUNK=8 python tests/t_chunk_equiv.py
"""
import os
import time

import torch

TEXT = ("昇腾处理器面向大规模深度学习训练与推理场景，其软件栈以CANN为核心，"
        "向上提供算子库、图引擎与分布式通信，向下驱动多代昇腾加速卡。"
        "推理侧关注的是首 token 延迟与稳定吞吐，训练侧关注的是集群扩展效率。")


def main():
    from fnx import engine as E
    from fnx.model import FnModel

    CH = E._CHUNK
    print(f"FNX_CHUNK={CH}", flush=True)
    m = FnModel(P=2048, lpc=45, dev0="npu:0", dev1="npu:1", nlayers=48,
                with_head=True, with_ngram=False, verbose=False)
    m.capture()
    tok = m.emb_q  # noqa: F841  仅为确认已加载
    from transformers import AutoTokenizer
    tk = AutoTokenizer.from_pretrained(os.environ.get("FNX_SRCDIR",
                                                      "/models/Qwen3.8-Flash-Next"))
    ids = tk(TEXT, return_tensors="pt").input_ids[0].tolist()
    print(f"prompt tokens={len(ids)}", flush=True)
    pre, last = ids[:-1], ids[-1]

    def run_pertoken():
        m.reset()
        for x in pre:
            m.step_pad(int(x), with_head=False)
        m.step_pad(int(last), with_head=True)
        return m.row_logits().float().clone()

    def run_chunk():
        m.reset()
        n = len(pre)
        i = 0
        while i + CH <= n:
            m.prefill_chunk(pre[i:i + CH])
            i += CH
        for x in pre[i:]:
            m.step_pad(int(x), with_head=False)
        m.step_pad(int(last), with_head=True)
        return m.row_logits().float().clone()

    lg_a = run_pertoken()
    lg_b = run_chunk()
    d = (lg_b - lg_a).abs()
    scale = lg_a.abs().mean().item() + 1e-9
    print(f"\n=== 数值 ===\n  logits maxdiff={d.max().item():.3e}  "
          f"mean={d.mean().item():.3e}  rel={d.mean().item()/scale:.3e}", flush=True)
    print(f"  argmax A={int(lg_a.argmax())}  B={int(lg_b.argmax())}  "
          f"{'一致' if int(lg_a.argmax()) == int(lg_b.argmax()) else '不一致'}", flush=True)

    def tm(fn, n=2, warm=1):
        for _ in range(warm):
            fn()
        torch.npu.synchronize()
        t = time.perf_counter()
        for _ in range(n):
            fn()
        torch.npu.synchronize()
        return (time.perf_counter() - t) / n

    t_a = tm(run_pertoken, n=2)
    t_b = tm(run_chunk, n=2)
    print(f"\n=== 速度（含 logits 的最后一步）===", flush=True)
    print(f"  逐 token : {t_a*1e3:8.1f} ms  → {len(ids)/t_a:6.1f} tok/s", flush=True)
    print(f"  分块 T={CH}: {t_b*1e3:8.1f} ms  → {len(ids)/t_b:6.1f} tok/s  "
          f"（{t_a/t_b:.2f}x）", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
