"""Aggregate r142 profiler traces into a closed time budget.

usage: analyze.py [runs.json] [traces_dir]
Decode windows are accounted per token period (between consecutive step ends);
prefill windows are a single chunk interval.
"""
import collections
import csv
import json
import sys
from decimal import Decimal
from pathlib import Path

P = Path(__file__).resolve().parent
RUNS = P / (sys.argv[1] if len(sys.argv) > 1 else 'runs.json')
TR = P / (sys.argv[2] if len(sys.argv) > 2 else 'traces')
OUT = P / (RUNS.stem + '-budget.json')
DEVICE = ('AI_CORE', 'AI_VECTOR_CORE', 'MIX_AIC', 'MIX_AIV')
CACHE = {}


def read(tag, name):
    key = (tag, name)
    if key in CACHE:
        return CACHE[key]
    hits = sorted(TR.glob(f'{tag}/**/{name}'))
    assert hits, (tag, name)
    CACHE[key] = list(csv.DictReader(hits[0].open()))
    return CACHE[key]


def num(s):
    s = (s or '').strip()
    return float(Decimal(s)) if s and s != 'N/A' else 0.0


def union(a):
    a = sorted((s, e) for s, e in a if e > s)
    if not a:
        return 0.0
    total = 0.0
    s, e = a[0]
    for x, y in a[1:]:
        if x <= e:
            e = max(e, y)
        else:
            total += e - s
            s, e = x, y
    return total + e - s


def group(n):
    if 'moe_cartesian_epilogue' in n:
        return 'MoE epilogue fused (r141)'
    if 'moe_merge' in n or 'weighted_reduce' in n:
        return 'HC/MoE fused reductions'
    if 'WeightQuant' in n:
        return 'W4/W8 quantized matmul'
    if 'GroupedMatmul' in n:
        return 'MoE grouped GMM'
    if 'MatMul' in n or 'BatchMatMul' in n:
        return 'Dense + index MatMul'
    if 'InferAttention' in n or 'FlashAttention' in n:
        return 'Sparse selected attention'
    if 'delta_sequential' in n:
        return 'FP32 DeltaNet'
    if 'RmsNorm' in n:
        return 'RMSNorm'
    if 'conv_prefill' in n or 'conv_four' in n:
        return 'Convolution'
    if any(x in n for x in ('gather_pair', 'expand_groups', 'write_dual_kv', 'residual_inject',
                            'hc_activation', 'hc_sigmoid_reduce', 'post_mul')):
        return 'Native fusion kernels'
    if any(x in n for x in ('Gather', 'Transpose', 'Slice', 'Concat', 'TensorMove', 'Scatter',
                            'IndexSelect', 'IndexCopy', 'Expand')):
        return 'Gather/layout/cache writes'
    return 'Other vector/scalar kernels'


def load_stage(tag, st):
    kernels = []
    for r in read(f'{tag}_stage{st}', 'kernel_details.csv'):
        s = num(r['Start Time(us)'])
        e = s + num(r['Duration(us)'])
        kernels.append((s, e, r['Name']))
    tasks = []
    for r in read(f'{tag}_stage{st}', 'task_time.csv'):
        kt = r['kernel_type'].strip().upper()
        if kt in DEVICE or 'MEMCPY' in kt:
            tasks.append((num(r['task_start(us)']), num(r['task_stop(us)']),
                          'MEMCPY' in kt))
    return kernels, tasks


def group_stats(rows, n):
    by = collections.defaultdict(lambda: dict(count=0, us=0.0))
    for s, e, name in rows:
        g = group(name)
        by[g]['count'] += 1
        by[g]['us'] += e - s
    return {k: dict(count=v['count'] / n, ms=v['us'] / 1000.0 / n)
            for k, v in sorted(by.items(), key=lambda kv: -kv[1]['us'])}


def analyze_decode(rec):
    tag = rec['tag']
    rows = rec['rows']
    edges = [r['realtime_end_ns'] / 1000.0 for r in rows]
    n = len(edges) - 1
    win = (edges[0], edges[-1])
    per_stage = {}
    kern_in = []
    for st in (0, 1):
        kernels, tasks = load_stage(tag, st)
        per_stage[st] = tasks
        kern_in += [(s, e, nm) for s, e, nm in kernels if s >= win[0] and e <= win[1]]
    dev = intra = outer = 0.0
    for i in range(1, len(edges)):
        a, b = edges[i - 1], edges[i]
        stage_dev = []
        stage_spans = []
        for st in (0, 1):
            ints = [(s, e) for s, e, _ in per_stage[st] if s >= a and e <= b]
            stage_dev.append(union(ints))
            dev += union(ints)
            if ints:
                stage_spans.append((min(s for s, e in ints), max(e for s, e in ints)))
        for st in (0, 1):
            ints = [(s, e) for s, e, _ in per_stage[st] if s >= a and e <= b]
            if ints:
                intra += (max(e for s, e in ints) - min(s for s, e in ints)) - union(ints)
        outer += (b - a) - union(stage_spans)
    period = sum(b - a for a, b in zip(edges, edges[1:])) / n
    out = dict(tag=tag, kind='decode', n_periods=n,
               unit_ms=period / 1000.0,
               kernel_device_ms=dev / n / 1000.0,
               intra_stage_gap_ms=intra / n / 1000.0,
               outer_host_gap_ms=outer / n / 1000.0,
               closure_ms=(dev + intra + outer) / n / 1000.0,
               groups=group_stats(kern_in, n),
               host_period_mean_ms=rec['mean_period_ms'],
               host_period_median_ms=rec['median_period_ms'])
    return out


def analyze_prefill(rec):
    tag = rec['tag']
    origin = rec['window'][0] / 1000.0
    width = (rec['window'][1] - rec['window'][0]) / 1000.0
    kern_in = []
    dev = 0.0
    stage_busy = []
    for st in (0, 1):
        kernels, tasks = load_stage(tag, st)
        kern_in += [(s - origin, e - origin, nm) for s, e, nm in kernels
                    if s >= origin and e <= origin + width]
        ints = [(s - origin, e - origin) for s, e, _ in tasks if s >= origin and e <= origin + width]
        stage_busy.append(union(ints) / 1000.0)
        dev += union(ints)
    return dict(tag=tag, kind='prefill_chunk', n_periods=1, chunk_size=rec.get('size'),
                chunk_offset=rec.get('offset'), unit_ms=width / 1000.0,
                kernel_device_ms=dev / 1000.0,
                idle_ms=(width - dev) / 1000.0,
                stage_busy_ms=stage_busy,
                groups=group_stats(kern_in, 1),
                wall_ms=rec.get('wall_ms'))


runs = json.loads(RUNS.read_text())
res = {}
for rec in runs:
    if not any(TR.glob(f"{rec['tag']}_stage0*")):
        res[rec['tag']] = dict(tag=rec['tag'], kind=rec['kind'] + '_unprofiled', n_periods=1,
                               unit_ms=rec.get('mean_period_ms'), groups={},
                               host_period_mean_ms=rec.get('mean_period_ms'),
                               host_period_median_ms=rec.get('median_period_ms'))
        continue
    res[rec['tag']] = analyze_decode(rec) if rec['kind'] == 'decode_window' else analyze_prefill(rec)
OUT.write_text(json.dumps(res, indent=2))

for tag, r in res.items():
    print(f"== {tag} ({r['kind']}) unit={r['unit_ms']:.3f} ms n={r['n_periods']}")
    if r.get('host_period_mean_ms'):
        print(f"   host period mean {r['host_period_mean_ms']:.3f} median {r['host_period_median_ms']:.3f} ms")
    if not r['groups']:
        continue
    if r['kind'] == 'decode':
        print(f"   device {r['kernel_device_ms']:.3f} + intra {r['intra_stage_gap_ms']:.3f} + "
              f"outer {r['outer_host_gap_ms']:.3f} = {r['closure_ms']:.3f}")
    else:
        print(f"   device {r['kernel_device_ms']:.3f} + idle {r['idle_ms']:.3f} = {r['unit_ms']:.3f} "
              f"(stage busy {[round(x, 1) for x in r['stage_busy_ms']]}, wall {r.get('wall_ms')})")
    for g, v in r['groups'].items():
        print(f"   {g:32s} n={v['count']:8.1f} ms={v['ms']:9.3f}")
