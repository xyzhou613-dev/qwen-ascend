"""t_chunk_hook: 给 FnLayer.forward / forward_chunk 打 hook，逐 token 逐层比输入输出。"""
import torch

IDS = [18643, 11111, 22222]


def fp(t):
    if isinstance(t, (list, tuple)):
        return [round(x.detach().float().sum().item(), 6) for x in t if torch.is_tensor(x)]
    return round(t.detach().float().sum().item(), 6)


def main():
    from fnx import engine as E
    from fnx.model import FnModel

    E._CHUNK = 1
    m = FnModel(P=2048, lpc=45, dev0="npu:0", dev1="npu:1", nlayers=48,
                with_head=True, with_ngram=False, verbose=False)
    L = E.FnLayer
    order = {id(l): i for i, l in enumerate(list(m.layers0) + list(m.layers1))}
    of, oc = L.forward, L.forward_chunk
    rec = {"base": [], "chunk": []}
    tag = ["base"]

    def wf(self, h, *a, **k):
        r = of(self, h, *a, **k)
        if tag[0] == "base":
            rec["base"].append((order[id(self)], fp(h), fp(r)))
        return r

    def wc(self, h, pos0, T):
        r = oc(self, h, pos0, T)
        if tag[0] == "chunk":
            rec["chunk"].append((order[id(self)], fp(h), fp(r)))
        return r

    L.forward, L.forward_chunk = wf, wc

    m.reset()
    for x in IDS:
        m.step_pad(x, with_head=False)
    tag[0] = "chunk"
    m.reset()
    for x in IDS:
        m.prefill_chunk([x])

    print("=== 逐 token 逐层 in/out 对比（A=forward, B=forward_chunk）===", flush=True)
    n = len(list(m.layers0)) + len(list(m.layers1))
    for t in range(len(IDS)):
        for i in range(n):
            a = rec["base"][t * n + i]
            b = rec["chunk"][t * n + i]
            ai, ao = a[1], a[2]
            bi, bo = b[1], b[2]
            if isinstance(ai, list):
                di = max(abs(x - y) for x, y in zip(ai, bi))
                do = max(abs(x - y) for x, y in zip(ao, bo))
            else:
                di = abs(ai - bi)
                do = abs(ao - bo)
            if di > 1e-5 or do > 1e-5:
                print(f"tok{t} L{i:02d}: in d={di:.3e}  out d={do:.3e}"
                      f"   A_in={ai}  B_in={bi}", flush=True)
                break
        else:
            print(f"tok{t}: 全部 48 层 in/out 一致", flush=True)
            continue
        break


if __name__ == "__main__":
    with torch.no_grad():
        main()
