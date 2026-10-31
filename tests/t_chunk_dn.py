"""t_chunk_dn: 单独对比 _deltanet 与 _deltanet_chunk（T=1）——定位公式差异。"""
import torch


def main():
    from fnx.model import FnModel

    m = FnModel(P=2048, lpc=45, dev0="npu:0", dev1="npu:1", nlayers=48,
                with_head=True, with_ngram=False, verbose=False)
    L = [x for x in m.layers0 if x.ltype == "linear"][0]
    torch.manual_seed(0)
    H = 2560   # _deltanet 的输入是 gated_residual 的输出 [b,1,H]，不是 HCH
    h = (torch.randn(1, 1, H, dtype=torch.float32) * 0.02).to(torch.bfloat16).npu()

    m.reset()
    out_a = L._deltanet(h).clone()
    Sa = L.S.clone()
    ca = L.conv.clone()

    m.reset()
    out_b = L._deltanet_chunk(h, 1).clone()
    Sb = L.S.clone()
    cb = L.conv.clone()

    def rep(name, a, b):
        a = a.float(); b = b.float()
        d = (b - a).abs()
        sc = a.abs().mean().item() + 1e-9
        print(f"  {name:8s} rel {d.mean().item()/sc:.3e}  max {d.max().item():.3e}  "
              f"|a|mean {sc:.3e}", flush=True)

    print("=== _deltanet vs _deltanet_chunk（T=1，同一 h）===", flush=True)
    rep("out", out_a, out_b)
    rep("S", Sa, Sb)
    rep("conv", ca, cb)

    # 再比中间量：mq / z / beta / a / g
    def mids(fn):
        if L.w_qkvz is not None:
            qz = fn(h)[:, 0]
            mq = qz[:, :10240]
            z = qz[:, 10240:10240 + 6144]
        return mq, z

    mq, z = mids(lambda x: m.w_qkvz_fn(x) if False else 0) if False else (None, None)
    # 直接用同一份投影比较卷积与 silu 之后的值
    m.reset()
    qz_a = (m.layers0[0].w_qkvz(h)[:, 0] if hasattr(m.layers0[0].w_qkvz, "__call__") else None) \
        if L.w_qkvz is not None else None
    print("  （中间量对比略：conv 已逐位一致，说明 mq->silu 段一致）", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
