"""t_wqbm_calls: 干净卡上量 npu_weight_quant_batchmatmul 的"每次调用固定开销 vs M/N 摊薄"。

为什么必须测：第 48 轮发现 MoE 每 token/层 1.33-1.44 ms，而分组后反而更慢。
若瓶颈是"每次调用的固定开销"，则优化方向不是减少权重流量，而是减少调用次数（或把 M 做大）。

用法（需先停 fnx.service，独占 npu:0）：
  bash /root/ascend-dev/p342_calls.sh
"""
import os
import time

import torch


def main():
    import torch_npu  # noqa: F401
    from fnx import load as L

    dev = torch.device("npu:0")
    with torch.npu.device(dev):
        W = L.load_layer(os.path.join(os.environ.get("FNX_QDIR",
                                                     "/models/Qwen3.8-Flash-Next-Q"),
                                     "L00.safetensors"), 0, dev)
        gu = W["mlp.experts.gate_up_proj"]          # K=2560 N=1280 E=512
        dn = W["mlp.experts.down_proj"]             # K=640  N=2560
        K1, N1, E_, G1 = gu.K, gu.N, gu.E, gu.G
        print(f"gu K={K1} N={N1} E={E_} G={G1};  dn K={dn.K} N={dn.N}", flush=True)

        gs = gu.gs
        gu_em = gu.w4.permute(1, 0, 2).contiguous()[0]      # 单专家 [K,N/8]（专家 0）
        sc1 = (gu.s8t[:, 0, :].to(torch.bfloat16) * gu.base[0].unsqueeze(0))   # [G,N]
        torch.npu.synchronize()

        def mk(N, k):
            """N 列、k 个专家沿 N 拼（权重用专家 0 重复，只为量开销）。"""
            if N <= N1:
                return gu_em[:, : N // 8].contiguous(), None
            w = gu_em.repeat(1, N // N1)                    # [K, N/8]
            s = sc1.repeat(1, N // N1)
            return w, s

        print("=== 单次调用成本（真实形状，E=512，expert0 权重） ===", flush=True)
        print(f"{'M':>5} {'N':>7} {'K':>6} {'ms/call':>9} {'GB/s':>8} {'us/Mrow':>9}", flush=True)
        for M in (1, 2, 4, 8, 16, 32, 64, 128):
            for N in (1280, 2560, 12800, 51200):
                if N == 1280:
                    w, s = gu_em, sc1
                elif N == 2560:
                    w, s = mk(2560, 2)
                elif N == 12800:
                    w, s = mk(12800, 10)
                else:
                    w, s = mk(51200, 40)
                x = torch.randn(M, K1, dtype=torch.bfloat16, device=dev) * 0.5
                fn = lambda: torch_npu.npu_weight_quant_batchmatmul(  # noqa: E731
                    x, w, s, antiquant_group_size=gs)
                fn()
                torch.npu.synchronize()
                t0 = time.time()
                n = 5
                for _ in range(n):
                    fn()
                torch.npu.synchronize()
                ms = (time.time() - t0) * 1000.0 / n
                gb = (w.numel() * 4) / 1e9
                print(f"{M:5d} {N:7d} {K1:6d} {ms:9.3f} {gb / (ms / 1e3):8.1f}"
                      f" {ms * 1e3 / M:9.1f}", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
