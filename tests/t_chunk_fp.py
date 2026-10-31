"""t_chunk_fp: 逐 token 对全部 48 层状态做指纹，定位第一处差异。"""
import torch

IDS = [18643, 11111, 22222, 33333, 44444, 55555, 66666, 77777]


def main():
    from fnx import engine as E
    from fnx.model import FnModel

    E._CHUNK = 1
    m = FnModel(P=2048, lpc=45, dev0="npu:0", dev1="npu:1", nlayers=48,
                with_head=True, with_ngram=False, verbose=False)
    layers = list(m.layers0) + list(m.layers1)

    def fp():
        d = {}

        def walk(tag, o):
            if isinstance(o, (list, tuple)):
                for j, c in enumerate(o):
                    walk(f"{tag}[{j}]", c)
            elif isinstance(o, torch.Tensor) and o.numel():
                d[tag] = o.detach().float().sum().item()

        for i, L in enumerate(layers):
            for name in ("S", "conv", "kv"):
                o = getattr(L, name, None)
                if o is not None:
                    walk(f"{i}.{name}", o)
        return d

    E._CHUNK = 1
    m.reset()
    ref = []
    for x in IDS:
        m.step_pad(x, with_head=False)
        ref.append(fp())

    m.reset()
    got = []
    for x in IDS:
        m.prefill_chunk([x])
        got.append(fp())

    print("=== 全层状态指纹对比（A=step, B=prefill_chunk）===", flush=True)
    for i, (a, b) in enumerate(zip(ref, got)):
        diffs = [(k, a[k], b[k]) for k in a
                 if abs(a[k] - b[k]) > 1e-6 * (abs(a[k]) + 1.0)]
        print(f"tok{i}: 差异项 {len(diffs)}/{len(a)}", flush=True)
        for k, va, vb in diffs[:12]:
            print(f"    {k:12s} A={va:+.8f}  B={vb:+.8f}  d={vb - va:+.3e}", flush=True)
        if diffs:
            break


if __name__ == "__main__":
    with torch.no_grad():
        main()
