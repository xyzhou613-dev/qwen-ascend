"""t_dense_w8_probe: 稠密权重三条路径对比（第 38 轮）。

(a) 现状：QW(x) = 每 token deq(整块 int8→bf16) + linear
(b) 预反量化：一次性 deq_w() 常驻 bf16，之后纯 linear（= FNX_DEQ_ATTN 做的事）
(c) 新路：npu_weight_quant_batchmatmul（W8 per-group，int8 + bf16 scale，Cube 内反量化）

同时验证数值，并给出“释放 bf16 预反量化副本能省多少显存”。
"""
import os
import time

import torch

DEV = "npu:0"
LAYER = int(os.environ.get("DW_LAYER", "12"))


def tm(fn, n=20, warm=4):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def main():
    import torch_npu
    from fnx import engine as E
    from fnx.load import load_layer

    path = os.environ.get("FNX_QDIR", "/models/Qwen3.8-Flash-Next-Q") + f"/L{LAYER:02d}.safetensors"
    W = load_layer(path, LAYER, DEV)
    names = [n for n in W if n.startswith(("linear_attn.", "self_attn.")) and
             not n.endswith(("conv1d.weight", "A_log", "dt_bias", "norm.weight",
                             "q_norm.weight", "k_norm.weight"))]
    print("layer %d 稠密权重：%s" % (LAYER, names), flush=True)
    f = torch_npu.npu_weight_quant_batchmatmul
    tot_cur = tot_pre = tot_new = 0.0
    for nm in sorted(names):
        w = W[nm]
        if not isinstance(w, E.QW):
            print("  %-22s 跳过（%s，已预反量化）" % (nm, type(w).__name__), flush=True)
            continue
        q, s, gs, K = w.q, w.s, w.gs, w.K          # q [N,K] int8, s [N,G] fp16
        N = q.shape[0]
        G = s.shape[-1]
        x = torch.randn(1, K, dtype=torch.bfloat16, device=DEV)
        y_ref = w(x).float()

        # (b) 预反量化
        bw = w.deq_w()                              # [N,K] bf16
        t_pre = tm(lambda: E._fl(x, bw))
        y_pre = E._fl(x, bw).float()

        # (c) aclnn W8 per-group：weight [K,N] int8 + scale [G,N] bf16
        w8t = q.t().contiguous()
        sc = s.t().contiguous().to(torch.bfloat16)
        ok = True
        try:
            y_new = f(x, w8t, sc, antiquant_group_size=gs).float()
            t_new = tm(lambda: f(x, w8t, sc, antiquant_group_size=gs))
        except Exception as e:  # noqa: BLE001
            ok = False
            print("  %-22s 算子 FAIL %s" % (nm, str(e)[:120]), flush=True)
            y_new = y_ref
            t_new = float("nan")
        t_cur = tm(lambda: w(x))

        tot_cur += t_cur
        tot_pre += t_pre
        tot_new += t_new if ok else 0.0
        rel_new = float((y_new - y_ref).abs().mean() / (y_ref.abs().mean() + 1e-9))
        rel_pre = float((y_pre - y_ref).abs().mean() / (y_ref.abs().mean() + 1e-9))
        print("  %-22s N=%5d K=%5d gs=%3d  cur %.3f  pre %.3f  new %.3f ms"
              "   rel_pre %.2e rel_new %.2e  mem(-int8+bf16scale) %+.1f MiB"
              % (nm, N, K, gs, t_cur, t_pre, t_new, rel_pre, rel_new,
                 -q.numel() / 2**20 + G * N * 2 / 2**20), flush=True)
        del bw, w8t, sc
    print("\n合计（一层 %d 个权重）: cur %.3f  pre %.3f  new %.3f ms"
          % (len(names), tot_cur, tot_pre, tot_new), flush=True)
    print("  换算 45 层: cur %.1f  pre %.1f  new %.1f ms/token"
          % (tot_cur * 45, tot_pre * 45, tot_new * 45), flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
