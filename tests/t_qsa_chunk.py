"""t_qsa_chunk: 单独对比 _qsa 与 _qsa_chunk（T=1）——定位数值差异。"""
import torch


def main():
    from fnx.model import FnModel

    m = FnModel(P=2048, lpc=45, dev0="npu:0", dev1="npu:1", nlayers=48,
                with_head=True, with_ngram=False, verbose=False)
    F = [x for x in m.layers0 if x.ltype == "full"][0]
    torch.manual_seed(0)
    h = (torch.randn(1, 1, 2560, dtype=torch.float32) * 0.02).to(torch.bfloat16).npu()

    m.reset()
    m.pos0.fill_(5)
    out_a = F._qsa(h, m.pos0).clone()
    ka = F.kv.k[F.ki].clone()
    va = F.kv.v[F.ki].clone()

    m.reset()
    out_b = F._qsa_chunk(h, 5, 1).clone()
    kb = F.kv.k[F.ki].clone()
    vb = F.kv.v[F.ki].clone()

    def rep(n, a, b):
        a = a.float()
        b = b.float()
        d = (b - a).abs()
        sc = a.abs().mean().item() + 1e-9
        print(f"  {n:6s} rel {d.mean().item()/sc:.3e}  max {d.max().item():.3e}", flush=True)

    print("=== _qsa vs _qsa_chunk (T=1, pos=5) ===", flush=True)
    rep("out", out_a, out_b)
    rep("k", ka, kb)
    rep("v", va, vb)


if __name__ == "__main__":
    with torch.no_grad():
        main()
