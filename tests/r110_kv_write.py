import json
import statistics
from types import SimpleNamespace

import torch
import torch_npu
from fnx.dual_write import write
from fnx.native_sparse import write_and_view

P = 32768
device = "npu:0"


def store():
    def tensor(*shape):
        # Sentinel verifies that rows outside the update stay untouched.
        return torch.full(shape, -3, device=device, dtype=torch.bfloat16)
    return SimpleNamespace(
        P=P, n=1, native_dual=True,
        k=[tensor(1, P, 2, 256).permute(0, 2, 1, 3), tensor(1, 2, P, 512)],
        v=[tensor(1, P, 2, 256).permute(0, 2, 1, 3), tensor(1, 2, P, 64)],
        head_offsets=torch.arange(2, device=device, dtype=torch.int64)[:, None] * P,
    )


def inputs(tokens, stride):
    data = torch.randint(-32768, 32768, (tokens, stride), dtype=torch.int32)
    return data.to(torch.int16).view(torch.bfloat16).to(device)


def baseline(dst, k, v, pos):
    write_and_view(dst, 0, k, v, pos, P, False)


def candidate(dst, k, v, pos):
    assert write(dst, 0, k, v, pos)


def compare(ref, out):
    for expected, got in zip(ref.k + ref.v, out.k + out.v):
        assert torch.equal(expected.view(torch.int16), got.view(torch.int16))


def graph(fn, dst, k, v, pos):
    g = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(g):
        fn(dst, k, v, pos)
    torch.npu.synchronize()
    return g


def measure(g):
    g.replay()
    torch.npu.synchronize()
    timings = []
    for _ in range(5):
        start, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(80):
            g.replay()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) * 1000 / 80)
    return statistics.median(timings)


def main():
    torch.manual_seed(1234)
    for tokens in (1, 17, 128, 512, 4096, 8192):
        # V can be a strided slice of the fused QKV projection.
        k = inputs(tokens, 512).reshape(tokens, 2, 256)
        v = inputs(tokens, 13312)[:, -512:].reshape(tokens, 2, 256)
        pos = torch.arange(tokens, device=device, dtype=torch.int64) + 17
        ref, out = store(), store()
        baseline(ref, k, v, pos)
        candidate(out, k, v, pos)
        compare(ref, out)
        old = graph(baseline, ref, k, v, pos)
        new = graph(candidate, out, k, v, pos)
        for start in (0, 2047, P - tokens):
            pos.copy_(torch.arange(tokens, device=device, dtype=torch.int64) + start)
            old.replay()
            new.replay()
            compare(ref, out)
        print(json.dumps(dict(tokens=tokens, exact=True, baseline_us=measure(old),
                              candidate_us=measure(new))), flush=True)


if __name__ == "__main__":
    main()
