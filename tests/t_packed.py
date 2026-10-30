"""★ 打包存储验证：FNX_PACKED_TORCH=0/1 两次运行，比 logits 与耗时/显存。

期望：
  - 数值：rel < 0.5%（两次 GEMM 求和顺序不同，属于 bf16 噪声级）
  - 显存：每层从 2.49 GiB 降到 ~1.36 GiB
  - 时间：每层多 ~1.46 ms（bf16 解包税）
"""
import os
import time

import torch

from fnx.model import FnModel

NL = int(os.environ.get("PK_NL", "8"))
NT = int(os.environ.get("PK_TOK", "6"))
TAG = os.environ.get("PK_TAG", "x")


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get("FNX_SRCDIR",
                                                       "/models/Qwen3.8-Flash-Next"))
    ids = tok("昇腾处理器面向大规模深度学习训练与推理场景", return_tensors="pt").input_ids[0, :NT].tolist()

    m = FnModel(P=2048, lpc=NL, dev0="npu:0", dev1="npu:1", nlayers=NL,
                with_head=True, with_ngram=False, verbose=False)
    packed = os.environ.get("FNX_PACKED_TORCH") == "1"
    print(f"=== packed={packed}  NL={NL} ===", flush=True)

    with torch.no_grad():
        m.reset()
        for i in range(3):
            m.step(int(ids[i]), with_head=True)
        m.reset()
        lgs = []
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for i in range(NT):
            m.step(int(ids[i]), with_head=True)
            lgs.append(m.logits.float().view(-1).clone())
        torch.npu.synchronize()
        dt = (time.perf_counter() - t0) / NT * 1e3

    nll = sum(float(-torch.log_softmax(lgs[i], -1)[ids[i + 1]]) for i in range(NT - 1)) / (NT - 1)
    print(f"  每步 {dt:8.2f} ms   mean_nll={nll:.4f}   "
          f"hbm0={torch.npu.memory_allocated(0)/2**30:.2f}GiB", flush=True)
    torch.save(torch.stack(lgs).cpu(), f"/tmp/pk_{TAG}.pt")


if __name__ == "__main__":
    with torch.no_grad():
        main()
