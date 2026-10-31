"""t_moe_break: 独占卡上把一层 MoE（B=1，生产口径）逐算子拆开，找出 1.33 ms/层 花在哪。

结论用途：干净卡上单次 WQBM 只要 ~0.05 ms，所以 MoE 的成本不可能是 GEMM 本身；
需要知道是 _wcat/_scat、topk/softmax、还是小算子数量（每个 ~50 µs 的固定成本）。
"""
import os
import time

import torch


def bench(fn, n=40):
    fn()
    torch.npu.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.time() - t0) * 1000.0 / n


def main():
    import torch_npu  # noqa: F401
    import torch.nn.functional as F
    from fnx import engine as E
    from fnx import load as L

    dev = torch.device("npu:0")
    with torch.npu.device(dev):
        W = L.load_layer(os.path.join(os.environ.get("FNX_QDIR",
                                                     "/models/Qwen3.8-Flash-Next-Q"),
                                     "L00.safetensors"), 0, dev)
        gu = W["mlp.experts.gate_up_proj"]
        dn = W["mlp.experts.down_proj"]
        H, MI, TOPK = E.H, E.MI, E.TOPK
        print(f"gu K={gu.K} N={gu.N} | dn K={dn.K} N={dn.N} | H={H} MI={MI} TOPK={TOPK}", flush=True)

        x = (torch.randn(1, H, device=dev) * 0.5).to(torch.bfloat16)
        ids = torch.topk(torch.rand(1, gu.E, device=dev), TOPK, dim=-1).indices[0]
        print("=== 逐算子（B=1，生产口径，单位 ms/次）===", flush=True)

        rows = []
        wg = gu._wcat(ids); sg = gu._scat(ids)
        wd = dn._wcat(ids); sd = dn._scat(ids)
        torch.npu.synchronize()

        rows.append(("_wcat(gu)  index_select 10 专家", bench(lambda: gu._wcat(ids))))
        rows.append(("_scat(gu)  int8->bf16 * base", bench(lambda: gu._scat(ids))))
        rows.append(("_wcat(dn)", bench(lambda: dn._wcat(ids))))
        rows.append(("_scat(dn)", bench(lambda: dn._scat(ids))))
        rows.append(("WQBM gu (M=1,N=12800,K=2560)",
                     bench(lambda: torch_npu.npu_weight_quant_batchmatmul(
                         x, wg, sg, antiquant_group_size=gu.gs))))
        y = torch_npu.npu_weight_quant_batchmatmul(x, wg, sg, antiquant_group_size=gu.gs)
        yv = y.view(TOPK, 2 * MI)
        rows.append(("silu*mul (act, 10x640)",
                     bench(lambda: (F.silu(yv[:, :MI].float()) * yv[:, MI:].float()).to(torch.bfloat16))))
        act = (F.silu(yv[:, :MI].float()) * yv[:, MI:].float()).to(torch.bfloat16)
        rows.append(("WQBM dn apply_diag (10x10 块取对角)",
                     bench(lambda: dn.apply_diag(act, ids))))

        logits = torch.randn(1, gu.E, device=dev)
        rows.append(("softmax(fp32) [1,512]", bench(lambda: torch.softmax(logits.float(), dim=-1))))
        p = torch.softmax(logits.float(), dim=-1)
        rows.append(("topk(10) [1,512]", bench(lambda: torch.topk(p, TOPK, dim=-1))))
        rows.append(("index_select 10 行 (embed 类小算子)",
                     bench(lambda: x.index_select(0, torch.zeros(10, dtype=torch.int64, device=dev)))))

        # 整块 _moe（用真实层对象）
        Lc = E.FnLayer
        for nm, v in rows:
            print(f"  {nm:42s} {v * 1e3:8.1f} us", flush=True)
        tot = sum(v for _, v in rows)
        print(f"  {'合计（若串行叠加）':42s} {tot * 1e3:8.1f} us = {tot:.3f} ms", flush=True)
        print(f"  生产实测 MoE ≈ 1.33 ms/层/token ⇒ 差额 {1.33 - tot:.2f} ms 在别处", flush=True)

        # 共享专家（DenseW8）的调用成本：M=1 的小 GEMM 有多少个、每个多久
        names = [k for k in W if "shared" in k or "gate" in k][:6]
        print("  共享专家/gate 权重:", names, flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
