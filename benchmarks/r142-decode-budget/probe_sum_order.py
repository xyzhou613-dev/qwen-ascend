"""Does a hand-written fp32 sum reproduce torch's (x*x).sum(-1) bitwise?"""
import torch
import torch_npu

torch.manual_seed(0)
dev = 'npu:0'
N = 128
x = torch.randn(64, 16, N, dtype=torch.float32, device=dev)
# realistic magnitudes: q/k after projection, bf16-rounded then fp32
xb = (torch.randn(64, 16, N, device=dev) * 3).to(torch.bfloat16).float()


def bits(t):
    return t.view(torch.int32)


def cmp(name, a, b):
    eq = (bits(a) == bits(b)).float().mean().item()
    print(f'{name:28s} exact={eq*100:.4f}%  maxulp={int((bits(a)-bits(b)).abs().max())}', flush=True)


for label, t in (('randn', x), ('bf16-rounded*3', xb)):
    ref = (t * t).sum(-1)
    # (a) sequential
    s = t[..., 0] * t[..., 0]
    for i in range(1, N):
        s = s + t[..., i] * t[..., i]
    cmp(label + ' seq', s, ref)
    # (b) 2 lanes strided
    for lanes in (2, 4, 8, 16, 32, 64):
        acc = [t[..., i] * t[..., i] for i in range(lanes)]
        for i in range(lanes, N, lanes):
            for j in range(lanes):
                acc[j] = acc[j] + t[..., i + j] * t[..., i + j]
        s = acc[0]
        for j in range(1, lanes):
            s = s + acc[j]
        cmp(label + f' lanes{lanes}', s, ref)
    # (c) pairwise tree
    v = [t[..., i] * t[..., i] for i in range(N)]
    while len(v) > 1:
        v = [v[i] + v[i + 1] for i in range(0, len(v), 2)]
    cmp(label + ' pairwise', v[0], ref)
    # (d) 8-wide block sequential (block sums then tree)
    blocks = []
    for b0 in range(0, N, 8):
        acc = t[..., b0] * t[..., b0]
        for i in range(b0 + 1, b0 + 8):
            acc = acc + t[..., i] * t[..., i]
        blocks.append(acc)
    v = blocks
    while len(v) > 1:
        v = [v[i] + v[i + 1] if i + 1 < len(v) else v[i] for i in range(0, len(v), 2)]
    cmp(label + ' block8+tree', v[0], ref)
print('PROBE_DONE', flush=True)
