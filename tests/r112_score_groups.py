import json
import math
import statistics

import torch
import torch_npu
from fnx.score import score

device = "npu:0"


def baseline(raw, pos):
    T, G = pos.numel(), raw.shape[1]
    scores = raw.view(T, 4, G).relu().sum(1) / math.sqrt(128)
    starts = torch.arange(G, device=device, dtype=torch.int64) * 4
    return scores.masked_fill(starts[None, :] + 4 > pos[:, None] + 1, -1.0e30)

def candidate(raw, pos):
    groups = raw.shape[1]
    starts = torch.arange(groups, device=device, dtype=torch.int64) * 4
    return (score(raw, pos) / math.sqrt(128)).masked_fill(
        starts[None, :] + 4 > pos[:, None] + 1, -1.0e30)


def measure(fn):
    g = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(g):
        result = fn()
    g.replay()
    torch.npu.synchronize()
    times = []
    for _ in range(5):
        start, end = [torch.npu.Event(enable_timing=True) for _ in range(2)]
        start.record()
        for _ in range(24):
            g.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) / 24)
    return statistics.median(times)


def main():
    torch.manual_seed(42)
    for T, G in ((1, 4096), (17, 4096), (512, 16384), (512, 65536)):
        # Dot product inputs, same [T * 4, G] FP32 layout as the indexer.
        q = torch.randn(T * 4, 128, device=device)
        k = torch.randn(128, G, device=device)
        raw = torch.matmul(q, k)
        pos = torch.arange(T, device=device, dtype=torch.int64) + G * 3 - 1
        ref = baseline(raw, pos)
        new = candidate(raw, pos)
        exact = torch.equal(ref.view(torch.int32), new.view(torch.int32))
        chosen_exact = torch.equal(ref.topk(512, -1).indices, new.topk(512, -1).indices)
        if not exact:
            a, b, c, d = raw.view(T, 4, G).relu().unbind(1)
            valid = (torch.arange(G, device=device)[None, :] + 1) * 4 <= pos[:, None] + 1
            sequential = (((a + b) + c) + d) / math.sqrt(128)
            paired = ((a + b) + (c + d)) / math.sqrt(128)
            print("ORDER", G, torch.equal(ref[valid], sequential[valid]),
                  torch.equal(ref[valid], paired[valid]), flush=True)
        print(json.dumps(dict(tokens=T, groups=G, exact=exact, selected=chosen_exact,
                              max_delta=(ref-new).abs().max().item(),
                              baseline_ms=measure(lambda: baseline(raw, pos)),
                              candidate_ms=measure(lambda: candidate(raw, pos)))), flush=True)
        assert exact and chosen_exact


if __name__ == "__main__":
    main()
