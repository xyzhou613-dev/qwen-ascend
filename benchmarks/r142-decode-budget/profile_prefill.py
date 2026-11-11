"""r142 pass 2: prefill chunk traces at production chunk sizes.

prefill8k  : 2048-row chunk at pos 6144 (production short-request policy)
prefillLate: last 8192-row chunk of a 262016-token prompt
"""
import json
import time
from pathlib import Path

import torch
import torch_npu
from fnx.server import Engine
from profile_support import make_profiler

OUT = Path(__file__).resolve().parent
SRC = Path('/models/fnx-diagnostics/r141-moe-epilogue-20261111')
TR = OUT / 'traces2'
METRIC = 'timing'


@torch.no_grad()
def main():
    e = Engine()
    m = e.m
    tok = e.tok
    pay = json.loads((SRC / 'baseline' / 'payload1.json').read_text())
    ids8k = tok(tok.apply_chat_template(pay['messages'], tokenize=False, add_generation_prompt=True,
                                        enable_thinking=False)).input_ids
    doc = Path('/root/ascend-dev/eval_text.txt').read_text(encoding='utf-8')
    unit = tok(doc + '\n').input_ids
    ids256 = (list(unit) * (262016 // len(unit) + 2))[:262016]
    print('PROMPTS_READY', len(ids8k), len(ids256), flush=True)
    runs = []
    print('CHUNK_GRAPHS', sorted(getattr(m.stage0, 'chunk_graphs', {}).keys()), flush=True)

    def csize(remaining):
        choices = [s for s in getattr(m.stage0, 'chunk_graphs', {}) if s <= remaining and s <= m.prefill_limit]
        size = max(choices) if choices else min(remaining, m.maximum, m.prefill_limit)
        if m.stage0._host_pos < 2048 and size > 512:
            size = min(512, remaining)
        return size

    def chunk_plan(ids):
        out = []
        off = 0
        vpos = m.stage0._host_pos
        while off < len(ids):
            remaining = len(ids) - off
            choices = [s for s in getattr(m.stage0, 'chunk_graphs', {}) if s <= remaining and s <= m.prefill_limit]
            size = max(choices) if choices else min(remaining, m.maximum, m.prefill_limit)
            if vpos < 2048 and size > 512:
                size = min(512, remaining)
            out.append((off, size))
            off += size
            vpos += size
        return out

    def start(tag):
        m._rpc('prof_start', str(TR / f'{tag}_stage1'), METRIC)
        p = make_profiler(str(TR / f'{tag}_stage0'), METRIC)
        p.start()
        return p

    def run(ids, tag, pick):
        m.plan_prefill(len(ids))
        plan = chunk_plan(ids)
        if pick == 'last8192':
            target = max((o, s) for o, s in plan if s == 8192)
        else:
            lim = int(pick)
            target = max(((o, s) for o, s in plan if s <= lim and o >= 2048), key=lambda x: x[1])
        prof = None
        t0 = w0 = None
        for off, size in plan:
            if (off, size) == target:
                prof = start(tag)
                t0 = time.time_ns()
                w0 = time.perf_counter_ns()
            m.prefill_chunk(ids[off:off + size])
            if (off, size) == target:
                m.finish_prefill()
                w1 = time.perf_counter_ns()
                t1 = time.time_ns()
                prof.stop()
                m._rpc('prof_stop')
                runs.append(dict(tag=tag, kind='prefill_chunk', offset=off, size=size,
                                 window=[t0, t1], wall_ms=(w1 - w0) / 1e6))
                print('PREFILL_CHUNK_DONE', tag, off, size, round((w1 - w0) / 1e6, 1), flush=True)
                prof = None
        m.finish_prefill()
        return plan

    m.reset()
    p8 = run(ids8k[:-1], 'prefill8k', '2048')
    print('P8_PLAN', len(p8), p8[:5], p8[-4:], flush=True)
    m.reset()
    p2 = run(ids256[:-1], 'prefillLate', 'last8192')
    print('P256_PLAN', len(p2), p2[:3], p2[-7:], flush=True)
    (OUT / 'runs2.json').write_text(json.dumps(runs, indent=2))
    print('R142_PREFILL_COMPLETE', flush=True)
    m.close()


if __name__ == '__main__':
    main()
