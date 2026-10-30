"""★ offload 机制验证：数值必须与不 offload 完全一致，并量出真实代价。

做法：8 层全放 dev0（lpc=8, nlayers=8），两种配置跑同一段 token：
  A 不 offload（FNX_OFFLOAD_FROM=999）
  B 从第 4 层起 offload（FNX_OFFLOAD_FROM=4）
比较每步 logits + 每步耗时 + 设备内存。

用 8 层是因为装配快（~2 分钟），且 4 层 offload 已能暴露所有问题。
"""
import os
import time

import torch

from fnx import engine as E
from fnx.model import FnModel

NL = int(os.environ.get("OF_NL", "8"))
NT = int(os.environ.get("OF_TOK", "6"))
OFF_FROM = int(os.environ.get("FNX_OFFLOAD_FROM", "999"))


def main():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(os.environ.get("FNX_SRCDIR",
                                                       "/models/Qwen3.8-Flash-Next"))
    ids = tok("昇腾处理器面向大规模深度学习训练与推理场景", return_tensors="pt").input_ids[0, :NT].tolist()

    m = FnModel(P=2048, lpc=NL, dev0="npu:0", dev1="npu:1", nlayers=NL,
                with_head=True, with_ngram=False, verbose=True)
    print(f"=== offload={OFF_FROM < NL}  FNX_OFFLOAD_FROM={OFF_FROM}  NL={NL} ===", flush=True)

    with torch.no_grad():
        m.reset()
        for i in range(3):                       # 预热
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
    torch.save({"logits": torch.stack(lgs).cpu(), "ids": ids},
               f"/tmp/off_{'on' if OFF_FROM < NL else 'off'}.pt")
    print(f"  已存 /tmp/off_{'on' if OFF_FROM < NL else 'off'}.pt", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
