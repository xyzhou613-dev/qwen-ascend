"""t_chunk_speed: 分块 prefill 的真实吞吐（eager 对 eager），并单独隔离 MoE 占比。"""
import os
import time

import torch

N = int(os.environ.get("SP_N", "256"))
SWEEP = [int(x) for x in os.environ.get("SP_SWEEP", "1,8,32,64,128").split(",")]


def run_one(m, chunks, ids, N):
    from fnx import engine as E
    E._CHUNK = chunks
    m.reset()
    if chunks == 1:
        for x in ids:
            m.step_pad(x, with_head=False)
    else:
        i = 0
        while i < N:
            m.prefill_chunk(ids[i:i + chunks])
            i += chunks
    torch.npu.synchronize()


def bench(m, chunks, ids, N):
    run_one(m, chunks, ids[:min(chunks, N)], min(chunks, N))     # 预热
    run_one(m, chunks, ids, N)                                   # 预热 2
    t0 = time.time()
    run_one(m, chunks, ids, N)
    return (time.time() - t0) * 1000.0 / N


def main():
    from fnx import engine as E
    from fnx.model import FnModel

    ids = [(18643 + 7 * i) % 60000 for i in range(N)]
    m = FnModel(P=2048, lpc=45, dev0="npu:0", dev1="npu:1", nlayers=48,
                with_head=True, with_ngram=False, verbose=False)
    print(f"=== 分块 prefill 吞吐（{N} token，eager 对 eager）===", flush=True)
    res = {}
    for ch in SWEEP:
        ms = bench(m, ch, ids, N)
        res[ch] = ms
        tag = "逐 token(step)" if ch == 1 else f"分块 {ch}"
        print(f"  {tag:12s} {ms:7.2f} ms/token   {1000.0 / ms:6.2f} tok/s", flush=True)

    H = E.H
    E.FnLayer._moe = lambda self, x: torch.zeros(x.shape[0], 1, H,
                                                dtype=x.dtype, device=x.device)
    print("=== 同一扫，但 MoE 置零（隔离稠密+注意力）===", flush=True)
    for ch in SWEEP:
        ms = bench(m, ch, ids, N)
        print(f"  chunk={ch:3d}  {ms:7.2f} ms/token   {1000.0 / ms:6.2f} tok/s"
              f"   MoE ≈ {res[ch] - ms:7.2f} ms/token", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
