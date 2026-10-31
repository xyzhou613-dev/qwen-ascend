"""t_chunk_driver: 直接对比 prefill_chunk([x]) 与 step_pad(x)，逐 token 看第一处偏差。"""
import torch

IDS = [18643, 11111, 22222, 33333, 44444, 55555, 66666, 77777]


def main():
    from fnx import engine as E
    from fnx.model import FnModel

    m = FnModel(P=2048, lpc=45, dev0="npu:0", dev1="npu:1", nlayers=48,
                with_head=True, with_ngram=False, verbose=False)
    L0 = m.layers0[0]
    F = [x for x in m.layers0 if x.ltype == "full"][0]

    def snap():
        return (L0.S.clone(), F.kv.k[F.ki].clone(), int(m.pos0[0]))

    E._CHUNK = 1
    m.reset()
    ref = []
    for x in IDS:
        m.step_pad(x, with_head=False)
        ref.append(snap())

    m.reset()
    got = []
    for x in IDS:
        m.prefill_chunk([x])
        got.append(snap())

    print("=== 逐 token 状态对比（A=step, B=prefill_chunk([x])）===", flush=True)
    for i, (a, b) in enumerate(zip(ref, got)):
        dS = (b[0].float() - a[0].float()).abs()
        rS = dS.mean().item() / (a[0].float().abs().mean().item() + 1e-9)
        dk = (b[1].float() - a[1].float()).abs()
        rk = dk.mean().item() / (a[1].float().abs().mean().item() + 1e-9)
        print(f"  tok{i}  S rel {rS:.3e}  kv rel {rk:.3e}  pos A={a[2]} B={b[2]}"
              f"{'   <-- 第一处偏差' if rS > 1e-6 and i == 0 else ''}", flush=True)
        if rS > 1e-6:
            break


if __name__ == "__main__":
    with torch.no_grad():
        main()
