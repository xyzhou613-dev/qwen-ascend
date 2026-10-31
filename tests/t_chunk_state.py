"""t_chunk_state: 修完归约后，检查 ①非零状态算子在位精确 ②端到端 prefill_chunk 与逐 token 逐位一致。"""
import torch

IDS = [18643, 11111, 22222, 33333, 44444, 55555, 66666, 77777]


def rel(a, b):
    d = (b.float() - a.float()).abs()
    return d.max().item(), (d.mean() / (a.float().abs().mean() + 1e-9)).item()


def main():
    from fnx import engine as E
    from fnx.model import FnModel

    E._CHUNK = 1
    m = FnModel(P=2048, lpc=45, dev0="npu:0", dev1="npu:1", nlayers=48,
                with_head=True, with_ngram=False, verbose=False)
    DN = m.layers0[0]
    FA = [x for x in m.layers0 if x.ltype == "full"][0]
    dev = DN.S.device

    m.reset()
    for x in IDS[:3]:
        m.step_pad(x, with_head=False)

    h = torch.randn(1, 1, 2560, dtype=torch.bfloat16, device=dev) * 0.1
    print("=== ① 非零状态 T=1 算子等价 ===", flush=True)
    S0, C0 = DN.S.clone(), DN.conv.clone()
    with torch.no_grad():
        y1 = DN._deltanet(h)
        S1, C1 = DN.S.clone(), DN.conv.clone()
        DN.S.copy_(S0); DN.conv.copy_(C0)
        y2 = DN._deltanet_chunk(h, 1)
        S2, C2 = DN.S.clone(), DN.conv.clone()
    for name, a, b in (("dn.y", y1, y2), ("dn.S", S1, S2), ("dn.conv", C1, C2)):
        mx, rl = rel(a, b)
        print(f"  {name:8s} max={mx:.3e} rel={rl:.3e}", flush=True)

    pos0 = 3
    with torch.no_grad():
        y1 = FA._qsa(h, torch.tensor([pos0], dtype=torch.int64, device=dev))
        K1 = FA.kv.k[FA.ki].clone()
        FA.kv.k[FA.ki].copy_(K1)
        y2 = FA._qsa_chunk(h, pos0, 1)
    mx, rl = rel(y1, y2)
    print(f"  qsa.y    max={mx:.3e} rel={rl:.3e}", flush=True)

    # ② 端到端：4 个 token 一次分块 vs 4 次逐 token
    print("=== ② 端到端（8 token，分块=1 与 4）===", flush=True)
    for CH in (1, 4, 8):
        E._CHUNK = 1
        m.reset()
        base = []
        for x in IDS:
            base.append(m.step_pad(x, with_head=False))
        S_b = DN.S.clone()
        m.reset()
        i = 0
        last = None
        while i < len(IDS):
            n = min(CH, len(IDS) - i)
            last = m.prefill_chunk(IDS[i:i + n])
            i += n
        mx, rl = rel(S_b, DN.S)
        print(f"  分块={CH}: L00.S max={mx:.3e} rel={rl:.3e}  pos0={int(m.pos0[0])}", flush=True)

    # ③ 只有 head 能看出 logits：单 token 对比
    print("=== ③ 单 token logits ===", flush=True)
    m.reset()
    l1 = m.step_pad(IDS[0], with_head=True)
    m.reset()
    l2 = m.prefill_chunk([IDS[0]])
    a = l1.float().flatten()
    b = l2.float().flatten()
    mx, rl = rel(a, b)
    print(f"  logits max={mx:.3e} rel={rl:.3e} argmax {a.argmax().item()} vs {b.argmax().item()}", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
