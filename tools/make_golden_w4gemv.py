#!/usr/bin/env python3
"""M1 黄金数据生成器：为 AscendC W4A16 GEMV 提供可逐元素比对的输入/期望输出。

产物目录 /root/ascend-dev/golden/w4gemv/：
  meta.json           形状、dtype、字节布局、容差
  x.bin               [1, K]      bf16  —— 单 token 激活
  gu_p4.bin           [N_gu, K/2] int8 —— gate_up 打包 4-bit（专家 e0）
  gu_s4.bin           [N_gu, K/32] fp16
  gu_y_ref.bin        [1, N_gu]  bf16 —— 期望输出（deq+bmm）
  dn_p4.bin / dn_s4.bin / dn_y_ref.bin   down_proj 同理（K=640）
另附 multi/：10 个专家一次的情形（M2 用）。

基准口径：用引擎里已验证过的 ExpQ.gather_deq（未打包 int8 路径）+ bmm，
与 kernel 的差异应在 bf16 噪声内（rel ≤1e-3）。
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/root/ascend-dev/fnx")
import torch_npu  # noqa
from fnx.load import load_layer, load_packed_experts, QDIR

OUT = "/root/ascend-dev/golden/w4gemv"
DEV = "npu:0"
EXPERT_IDS = [7, 343, 105, 212, 243, 254, 403, 168, 302, 67]   # L00 真实路由（第 25 轮实测）


def dump(name, t):
    p = os.path.join(OUT, name)
    c = t.detach().cpu().contiguous()
    if c.dtype == torch.bfloat16:
        c = c.view(torch.int16)      # bf16 原始位型，numpy 不支持 bf16
    c.numpy().tofile(p)
    return {"file": name, "shape": list(t.shape), "dtype": ("bf16_raw_int16" if t.dtype == torch.bfloat16 else str(t.dtype).replace("torch.", "")),
            "nbytes": os.path.getsize(p)}


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(os.path.join(OUT, "multi"), exist_ok=True)
    torch.manual_seed(0)

    W = load_layer(f"{QDIR}/L00.safetensors", 0, DEV)      # 未打包 int8 路径 = 基准
    pk = load_packed_experts(0, DEV)                       # 落盘真 4-bit = kernel 输入
    meta = {"layer": 0, "expert_ids": EXPERT_IDS,
            "contract": {
                "pack_axis": "last (K), byte j: low4=k=2j, high4=k=2j+1",
                "nibble": "two's complement in 0..15, value=((n^8)-8), range -8..7",
                "dequant": "W[n,k] = nibble(n,k) * s4[n, k//32]",
                "op": "y[0,n] = sum_k W[n,k] * x[0,k]",
            },
            "tolerance": {"rel_mean": 1e-3, "rel_max": 5e-3},
            "cases": {}}

    for which, key, N, K in (("gate_up_proj", "mlp.experts.gate_up_proj", 1280, 2560),
                             ("down_proj", "mlp.experts.down_proj", 2560, 640)):
        e0 = EXPERT_IDS[0]
        x = (torch.randn(1, K, dtype=torch.bfloat16, device=DEV) / 16)
        ref = W[key].gather_deq(torch.tensor([e0], device=DEV)).float()      # [1,N,K]
        y = torch.bmm(x.float().unsqueeze(1), ref.transpose(1, 2)).squeeze(1)  # [1,N]
        p4, s4 = pk[which]
        d = {
            "x": dump(f"{which}_x.bin", x),
            "p4": dump(f"{which}_p4.bin", p4[e0]),
            "s4": dump(f"{which}_s4.bin", s4[e0]),
            "y_ref": dump(f"{which}_y_ref.bin", y.to(torch.bfloat16)),
            "N": N, "K": K, "gs": 32, "expert": e0,
            "y_absmean": float(y.abs().mean()),
        }
        meta["cases"][which] = d
        print(f"  {which}: N={N} K={K}  p4 {p4[e0].numel()/2**20:.1f}MiB  "
              f"y[{N}] |y|mean={d['y_absmean']:.4f}", flush=True)

    # M2：10 专家一次发射
    for which, key, N, K in (("gate_up_proj", "mlp.experts.gate_up_proj", 1280, 2560),
                             ("down_proj", "mlp.experts.down_proj", 2560, 640)):
        idx = torch.tensor(EXPERT_IDS, device=DEV)
        x = (torch.randn(len(EXPERT_IDS), K, dtype=torch.bfloat16, device=DEV) / 16)
        ref = W[key].gather_deq(idx).float()
        y = torch.bmm(x.float().unsqueeze(1), ref.transpose(1, 2)).squeeze(1)
        p4, s4 = pk[which]
        sub = os.path.join(OUT, "multi")
        p4.index_select(0, idx).cpu().contiguous().numpy().tofile(
            os.path.join(sub, f"{which}_p4.bin"))
        s4.index_select(0, idx).cpu().contiguous().numpy().tofile(
            os.path.join(sub, f"{which}_s4.bin"))
        x.cpu().contiguous().view(torch.int16).numpy().tofile(os.path.join(sub, f"{which}_x.bin"))
        y.to(torch.bfloat16).cpu().contiguous().view(torch.int16).numpy().tofile(
            os.path.join(sub, f"{which}_y_ref.bin"))
        meta["cases"][f"multi_{which}"] = {
            "dir": "multi", "B": len(EXPERT_IDS), "N": N, "K": K, "gs": 32}

    json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
    tot = sum(os.path.getsize(os.path.join(dp, f))
              for dp, _, fs in os.walk(OUT) for f in fs)
    print(f"\n=== 黄金数据 {OUT}  {tot/2**20:.1f} MiB ===")
    print("单专家比对：python tools/cmp_w4gemv.py <which> <kernel_out.bin>")


if __name__ == "__main__":
    main()
