"""t_chunk_layer: 单层 forward vs forward_chunk（T=1）——最后一层定位。"""
import torch


def main():
    from fnx.model import FnModel

    m = FnModel(P=2048, lpc=45, dev0="npu:0", dev1="npu:1", nlayers=48,
                with_head=True, with_ngram=False, verbose=False)
    L0 = [x for x in m.layers0 if x.ltype == "linear"][0]
    L3 = [x for x in m.layers0 if x.ltype == "full"][0]
    torch.manual_seed(0)
    h = (torch.randn(1, 1, 10240, dtype=torch.float32) * 0.02).to(torch.bfloat16).npu()

    def rep(n, a, b):
        a = a.float()
        b = b.float()
        d = (b - a).abs()
        sc = a.abs().mean().item() + 1e-9
        print(f"  {n:22s} rel {d.mean().item()/sc:.3e}  max {d.max().item():.3e}", flush=True)

    for L in (L0, L3):
        m.reset()
        m.pos0.fill_(3)
        out_a = L.forward(h, m.pos0).clone()
        Sa = L.S.clone() if hasattr(L, "S") else None
        ka = L.kv.k[L.ki].clone() if L.kv is not None else None

        m.reset()
        out_b = L.forward_chunk(h, 3, 1).clone()
        Sb = L.S.clone() if hasattr(L, "S") else None
        kb = L.kv.k[L.ki].clone() if L.kv is not None else None

        print(f"=== 层 L{L.idx} ({L.ltype}) forward vs forward_chunk(T=1) ===", flush=True)
        rep("out", out_a, out_b)
        if Sa is not None:
            rep("S", Sa, Sb)
        if ka is not None:
            rep("kv.k", ka, kb)


if __name__ == "__main__":
    with torch.no_grad():
        main()
