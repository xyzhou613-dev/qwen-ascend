"""t_moe_grp2: 独占 npu:0 上重测 MoE 三条路径（修正版：显式设置当前 device）。

A) 逐 token：生产路径（每 token 拷 10 个专家的 int4 权重 → 一次 WQBM）
B) 按专家分组 + E-major 布局（|U|≈470 次单专家调用，无拷贝）
C) 按专家分组 + 现有 [K,E,N/8] 布局（每专家先 contiguous()）

旧版 t_moe_grp.py 跑在 npu:1（与生产共享、且未设当前 device），单次调用被放大到 0.63 ms，
本版在独占卡上复核。
"""
import os
import time

import torch


def main():
    import torch_npu  # noqa: F401
    from fnx import engine as E
    from fnx import load as L

    dev = torch.device("npu:0")
    T = int(os.environ.get("T", "128"))
    with torch.npu.device(dev):
        qdir = os.environ.get("FNX_QDIR", "/models/Qwen3.8-Flash-Next-Q")
        W = L.load_layer(os.path.join(qdir, "L00.safetensors"), 0, dev)
        gu = W["mlp.experts.gate_up_proj"]
        dn = W["mlp.experts.down_proj"]
        print(f"gu: K={gu.K} N={gu.N} E={gu.E} G={gu.G} gs={gu.gs} w4={tuple(gu.w4.shape)}", flush=True)
        print(f"dn: K={dn.K} N={dn.N} E={dn.E} G={dn.G} w4={tuple(dn.w4.shape)}", flush=True)

        H, EN, MI, TOPK = E.H, gu.E, E.MI, E.TOPK
        torch.npu.synchronize()

        t0 = time.time()
        gu_em = gu.w4.permute(1, 0, 2).contiguous()
        dn_em = dn.w4.permute(1, 0, 2).contiguous()
        torch.npu.synchronize()
        print(f"E-major 翻转 {time.time() - t0:.2f}s gu {tuple(gu_em.shape)}", flush=True)

        torch.manual_seed(0)
        x = (torch.randn(T, H, device=dev, dtype=torch.float32) * 0.5).to(torch.bfloat16)
        idx = torch.topk(torch.rand(T, EN, device=dev), TOPK, dim=-1).indices
        wt = torch.softmax(torch.randn(T, TOPK, device=dev), dim=-1).to(torch.bfloat16)
        wt = wt / wt.sum(-1, keepdim=True)

        order = torch.argsort(idx.reshape(-1))
        cnt = torch.bincount(idx.reshape(-1)[order], minlength=EN)
        off = torch.cumsum(cnt, 0) - cnt
        tok = order // TOPK
        wflat = wt.reshape(-1).float()
        nz = torch.nonzero(cnt, as_tuple=False).flatten().tolist()
        u = len(nz)
        print(f"T={T} 每 token {TOPK} 专家 -> 并集 {u}/{EN}；每 token 权重 {TOPK * 3.28:.1f} MB"
              f" / 分组后 {u * 3.28 / T:.1f} MB", flush=True)

        t0 = time.time()
        sg = {e_: (gu.s8t[:, e_, :].to(torch.bfloat16) * gu.base[e_].unsqueeze(0)) for e_ in nz}
        sd = {e_: (dn.s8t[:, e_, :].to(torch.bfloat16) * dn.base[e_].unsqueeze(0)) for e_ in nz}
        torch.npu.synchronize()
        print(f"scale 面板预热 {time.time() - t0:.2f}s", flush=True)

        def call(xin, w2d, sc, src):
            return torch_npu.npu_weight_quant_batchmatmul(
                xin, w2d, sc, antiquant_group_size=src.gs)

        def act_of(y):
            yv = y.view(-1, 2 * MI) if y.dim() == 2 else y
            return (torch.nn.functional.silu(yv[:, :MI].float()) * yv[:, MI:].float()).to(torch.bfloat16)

        def path_a():
            outs = []
            for t in range(T):
                y = gu.apply(x[t:t + 1], idx[t]).view(TOPK, 2 * MI)
                act = (torch.nn.functional.silu(y[:, :MI].float()) * y[:, MI:].float()).to(torch.bfloat16)
                y2 = dn.apply_diag(act, idx[t])
                outs.append((y2 * wt[t].unsqueeze(-1)).sum(0, keepdim=True))
            return torch.cat(outs, 0)

        def path_grp(em_major):
            acc = torch.zeros(T, H, dtype=torch.float32, device=dev)
            for e_ in nz:
                o = int(off[e_]); c = int(cnt[e_])
                rows = order[o:o + c]
                tk = tok[rows]
                xg = x.index_select(0, tk)
                if em_major:
                    wg2, wd2 = gu_em[e_], dn_em[e_]
                else:
                    wg2 = gu.w4[:, e_, :].contiguous()
                    wd2 = dn.w4[:, e_, :].contiguous()
                yg = call(xg, wg2, sg[e_], gu)
                actg = act_of(yg)
                yd = call(actg, wd2, sd[e_], dn)
                acc.index_add_(0, tk, yd.float() * wflat[rows].unsqueeze(-1))
            return acc.to(torch.bfloat16)

        def bench(fn, n, name):
            fn()
            torch.npu.synchronize()
            t0 = time.time()
            out = None
            for _ in range(n):
                out = fn()
            torch.npu.synchronize()
            ms = (time.time() - t0) * 1000.0 / n
            print(f"  {name:30s} {ms:8.2f} ms / {T} tok = {ms / T * 1e3:7.1f} us/tok"
                  f" = {ms / T:6.2f} ms/token", flush=True)
            return out, ms

        print(f"=== 计时（真实形状 L00，E=512，T={T}）===", flush=True)
        outa, msa = bench(path_a, 2, "A 逐 token（生产路径）")
        outb, msb = bench(lambda: path_grp(True), 2, "B 分组 + E-major")
        outc, msc = bench(lambda: path_grp(False), 1, "C 分组 + 现有布局 contiguous()")

        for nm, o in (("B", outb), ("C", outc)):
            d = (o.float() - outa.float()).abs()
            print(f"  数值 {nm} vs A: max={d.max().item():.3e} "
                  f"rel={(d.mean() / (outa.float().abs().mean() + 1e-9)).item():.3e}", flush=True)
        print(f"  => 分组/生产 = {msa / msb:.2f}x (E-major) / {msa / msc:.2f}x (contiguous)", flush=True)

        t0 = time.time()
        for t in range(16):
            gu._wcat(idx[t])
        torch.npu.synchronize()
        cms = (time.time() - t0) * 1000.0 / 16
        print(f"  仅 _wcat（index_select 10 专家权重）= {cms:.3f} ms/次"
              f"（gu+dn 约 {cms * 2:.2f} ms/token/layer）", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
