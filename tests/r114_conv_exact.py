import os
"""Check convolution rounding, evolving history, and captured replay."""
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu
from fnx.conv_native import step


def reference(history, inputs, weights):
    buf = torch.cat((history, inputs.unsqueeze(-1)), -1)
    output = (buf * weights.unsqueeze(0)).sum(-1)
    history.copy_(buf[..., 1:])
    return output


def equal(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def captured(fn, repetitions=1):
    graph = torch.npu.NPUGraph()
    torch.npu.synchronize()
    with torch.npu.graph(graph):
        for _ in range(repetitions):
            output = fn()
    torch.npu.synchronize()
    return graph, output


def timing(fn):
    graph, output = captured(fn, 36)
    graph.replay()
    torch.npu.synchronize()
    times = []
    for _ in range(12):
        start = time.perf_counter()
        graph.replay()
        torch.npu.synchronize()
        times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times[2:])


@torch.no_grad()
def main():
    torch.manual_seed(114)
    D = 10240
    device = "npu:0"
    h0 = torch.randn(1, D, 3, device=device, dtype=torch.bfloat16)
    x = torch.randn(1, D, device=device, dtype=torch.bfloat16)
    weights = torch.randn(D, 4, device=device, dtype=torch.bfloat16)
    # Cancellation cases distinguish FP32 reduction trees.
    h0[0, 0] = torch.tensor([1e8, 1, -1e8], dtype=torch.bfloat16, device=device)
    h0[0, 1] = torch.tensor([1e8, -1e8, 1], dtype=torch.bfloat16, device=device)
    x[0, :2] = 1
    weights[:2] = 1
    rows = []
    for order in (0, 1, 2):
        history = h0.clone()
        ref_history = h0.clone()
        actual = step(history, x, weights, order)
        expected = reference(ref_history, x, weights)
        torch.npu.synchronize()
        row = dict(order=order, output_exact=equal(actual, expected),
                   history_exact=equal(history, ref_history),
                   max_delta=(actual.float() - expected.float()).abs().max().item())
        rows.append(row)
        print("CONV_ORDER", json.dumps(row), flush=True)
    valid = [r["order"] for r in rows if r["output_exact"] and r["history_exact"]]
    assert valid, rows
    order = valid[0]
    history = h0.clone()
    expected_history = h0.clone()
    graph, output = captured(lambda: step(history, x, weights, order))
    history.copy_(h0)
    for i in range(256):
        x.copy_(torch.randn_like(x) * (.1 if i % 2 else 2))
        graph.replay()
        expected = reference(expected_history, x, weights)
        torch.npu.synchronize()
        assert equal(output, expected), ("output", i)
        assert equal(history, expected_history), ("history", i)
    times = {}
    for name, fn in (
        ("reference", lambda: torch.nn.functional.silu(reference(history, x, weights))),
        ("native", lambda: torch.nn.functional.silu(step(history, x, weights, order))),
        ("reference_repeat", lambda: torch.nn.functional.silu(reference(history, x, weights))),
    ):
        times[name] = timing(fn)
    result = dict(order=order, recurrent_steps=256, exact=True, samples=rows, graph36_ms=times)
    Path(os.environ.get("FNX_CONV_TEST_OUTPUT", "/tmp/fnx-conv-test.json")).write_text(json.dumps(result, indent=2))
    print("R114_CONV_ACCEPT", json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
