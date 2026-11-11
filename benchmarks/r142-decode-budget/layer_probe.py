"""Real L00 decode: deployed gather vs double-buffered gather, bitwise."""
import ctypes
import json
import statistics
import sys
from pathlib import Path

import torch
import torch_npu

P = Path('/models/fnx-diagnostics/r142-budget-20261111')
OUT = P
SIG = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]


def load_lib(name):
    lib = ctypes.CDLL(str(P / name))
    lib.fnx_gather_pair.argtypes = SIG
    lib.fnx_gather_pair.restype = ctypes.c_int
    return lib


@torch.no_grad()
def main():
    from fnx import engine as E
    from fnx.load import load_layer, QDIR
    import fnx.expert_gather as EG

    torch.npu.set_device(0)
    torch.manual_seed(4242)
    weights = load_layer(str(Path(QDIR) / 'L00.safetensors'), 0, 'npu:0')
    L = E.FnLayer(0, 'linear', weights, 'npu:0')
    pos = torch.tensor([5], device='npu:0')
    rows = []
    states = {}
    for mode, so in (('orig', 'libfnx_expert_gather_deployed.so'), ('db', 'libfnx_expert_gather_db.so')):
        EG._lib = load_lib(so)
        torch.manual_seed(99)
        S0 = (torch.randn_like(L.S) * 0.01).contiguous()
        C0 = (torch.randn_like(L.conv) * 0.03).contiguous()
        outs = []
        for _ in range(8):
            x = torch.randn((1, 1, E.HCH), device='npu:0', dtype=torch.bfloat16) * 0.03
            L.S.copy_(S0)
            L.conv.copy_(C0)
            y = L.forward(x, pos)
            outs.append((y.clone(), L.S.clone(), L.conv.clone()))
        states[mode] = outs
    acc = dict(output_exact=True, state_exact=True, conv_exact=True,
               output_max_abs=0.0, state_max_abs=0.0)
    for (a, sa, ca), (b, sb, cb) in zip(states['orig'], states['db']):
        acc['output_exact'] &= torch.equal(a.view(torch.int16), b.view(torch.int16))
        acc['state_exact'] &= torch.equal(sa.view(torch.int32), sb.view(torch.int32))
        acc['conv_exact'] &= torch.equal(ca.view(torch.int16), cb.view(torch.int16))
        acc['output_max_abs'] = max(acc['output_max_abs'], (a.float() - b.float()).abs().max().item())
        acc['state_max_abs'] = max(acc['state_max_abs'], (sa.float() - sb.float()).abs().max().item())

    timing = {}
    graphs = {}
    for mode, so in (('orig', 'libfnx_expert_gather_deployed.so'), ('db', 'libfnx_expert_gather_db.so')):
        EG._lib = load_lib(so)
        x = torch.randn((1, 1, E.HCH), device='npu:0', dtype=torch.bfloat16) * 0.03
        L.forward(x, pos)
        torch.npu.synchronize()
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            L.forward(x, pos)
        graphs[mode] = (g, x)
    for _ in range(6):
        for mode in ('orig', 'db', 'orig', 'db'):
            g, x = graphs[mode]
            samples = []
            for _ in range(5):
                a = torch.npu.Event(enable_timing=True)
                b = torch.npu.Event(enable_timing=True)
                a.record()
                for _ in range(16):
                    g.replay()
                b.record()
                b.synchronize()
                samples.append(a.elapsed_time(b) / 16)
            timing.setdefault(mode, []).append(statistics.median(samples))
    rows.append(dict(accuracy=acc, timing={k: statistics.median(v) for k, v in timing.items()}))
    (OUT / 'layer-gather.json').write_text(json.dumps(rows, indent=2))
    print('LAYER_GATHER', json.dumps(rows[-1]), flush=True)
    print('LAYER_GATHER_EXACT' if (acc['output_exact'] and acc['state_exact'] and acc['conv_exact'])
          else 'LAYER_GATHER_MISMATCH', flush=True)


if __name__ == '__main__':
    main()
