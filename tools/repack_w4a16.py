"""把 LQP artifact 的专家权重转成 aclnn per-group W4A16 要的形态。

为什么：`torch_npu.npu_weight_quant_batchmatmul`（aclnnWeightQuantBatchMatmulV2）在
Atlas A2 上支持 per-group W4A16，解包/乘 scale/GEMM 全在 AI Core 内完成。
实测（第 37 轮，真实形状 K=2560->N=1280, 10 专家拼接）：一次调用 0.043 ms，
而当前 torch 路径（16.4M 元素解包 + 2 次 bmm）约 1.0 ms/层。

产出（每层一个文件，只含专家张量）：
  mlp.experts.<proj>.w4   int32 [K, E, N/8]   —— npu_convert_weight_to_int4pack 的结果，
                             **转置成 [K,E,N/8]**：这样 index_select(dim=1) 出来的就是
                             [K, k, N/8] 连续布局，可直接 reshape 成 [K, k*N/8] 给算子，省掉一次
                             16.4 MB 的 permute（实测 0.068 -> 0.036 ms/层）。
  mlp.experts.<proj>.s8t  int8  [G, E, N]     —— 与 v2o 里的 s8 逐位相同，只是转置；
                             per-group 模式算子要求 scale 形状是 (ceil(K,gs), N)。
  mlp.experts.<proj>.base bf16  [E, N]        —— 必须 bf16（算子要求 scale 与 x 同 dtype）

数值：不解量化、不重新量化，只是把同一个 int4 值换个容器 + 换布局。scale 仍是 g64 的 int8
+ 每行 fp16 base（转成 bf16 只是为了喂算子；bf16 有 8 位尾数，与 fp16 的 scale 值量级
（1e-3~1e-2）匹配，误差远低于 4-bit 本身的量化误差）。

用法（容器内）：python tools/repack_w4a16.py [层数]
"""
import json
import os
import sys
import time

import numpy as np
import torch

SRC = os.environ.get("FNX_LQP", "/models/Qwen3.8-Flash-Next-LQP")
DST = os.environ.get("FNX_W4A16_DIR", "/models/Qwen3.8-Flash-Next-LQP-w4a16")
DEV = "npu:0"
CHUNK = 32          # 每次处理多少个专家（控宿主内存）

import torch_npu  # noqa: E402

pack = torch_npu.npu_convert_weight_to_int4pack


def unpack_v2(v2o, K):
    """[m,N,K/2] uint8（offset-binary v2 布局）-> [m,N,K] int32，值域 -8..7。

    布局（engine.ExpQ._deq_v2 的逆）：byte(64s+j) 的低 4 bit = W[128s+j]，
    高 4 bit = W[128s+64+j]；n' = n + 8。
    """
    m, N, _ = v2o.shape
    S = K // 128
    p = v2o.astype(np.int16)
    hi = (p >> 4) - 8
    lo = (p & 15) - 8
    out = np.empty((m, N, S, 2, 64), dtype=np.int32)
    out[..., 0, :] = lo.reshape(m, N, S, 64)
    out[..., 1, :] = hi.reshape(m, N, S, 64)
    return out.reshape(m, N, K)


def conv_proj(sd, proj, E, K, N, G):
    t0 = time.perf_counter()
    v2o = sd[f"mlp.experts.{proj}.v2o"].numpy()          # [E,N,K/2] uint8
    s8 = sd[f"mlp.experts.{proj}.s8"]                    # [E,N,G] int8
    base = sd[f"mlp.experts.{proj}.base"]                # [E,N] fp16
    assert v2o.shape == (E, N, K // 2), (v2o.shape, E, N, K)
    art = torch.empty(K, E, N // 8, dtype=torch.int32, device=DEV)
    for e0 in range(0, E, CHUNK):
        e1 = min(e0 + CHUNK, E)
        q = unpack_v2(v2o[e0:e1], K)                     # [m,N,K] int32
        for i in range(e1 - e0):
            qt = torch.from_numpy(q[i].T.copy()).to(DEV)  # [K,N] int32（宿主转置一次）
            art[:, e0 + i, :] = pack(qt)
        del q
    s8t = s8.permute(2, 0, 1).contiguous()                # [G,E,N]
    bf = base.to(torch.bfloat16)
    print(f"    {proj}: {time.perf_counter()-t0:.1f}s  art {tuple(art.shape)} "
          f"{art.numel()*4/2**20:.0f} MB", flush=True)
    return art.cpu(), s8t, bf


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 48
    os.makedirs(DST, exist_ok=True)
    from safetensors.torch import load_file, save_file
    for L in range(n):
        p = os.path.join(SRC, f"L{L:02d}.safetensors")
        if not os.path.exists(p):
            print(f"[skip] {p}", flush=True)
            continue
        out = os.path.join(DST, f"L{L:02d}.w4.safetensors")
        if os.path.exists(out):
            print(f"[skip done] {out}", flush=True)
            continue
        t0 = time.perf_counter()
        print(f"[L{L:02d}] loading {p}", flush=True)
        sd = load_file(p, device="cpu")
        res = {}
        for proj, (K, N, G) in (("gate_up_proj", (2560, 1280, 40)), ("down_proj", (640, 2560, 10))):
            w4, s8t, bf = conv_proj(sd, proj, 512, K, N, G)
            res[f"mlp.experts.{proj}.w4"] = w4
            res[f"mlp.experts.{proj}.s8t"] = s8t
            res[f"mlp.experts.{proj}.base"] = bf
        del sd
        save_file(res, out)
        print(f"[L{L:02d}] done {time.perf_counter()-t0:.1f}s -> {out} "
              f"({os.path.getsize(out)/2**30:.2f} GiB)", flush=True)
        del res
    with open(os.path.join(DST, "manifest.json"), "w") as f:
        json.dump({"source": SRC, "layout": "w4=[K,E,N/8] int32 int4pack; s8t=[G,E,N] int8; base=[E,N] bf16",
                   "gs": 64, "layers": n}, f, indent=1)
    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
