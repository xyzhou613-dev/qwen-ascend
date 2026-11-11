"""r142 timing evidence on production r141 code.

Passes:
  decode8k    : 8064-token prompt, steady-state decode near 8k
  prefill8k   : one 2048-token prefill chunk at pos 4096 (short-request policy)
  prefill0    : first 8192-token chunk of a 262016-token prompt
  prefillLate : 8192-token chunk at pos 245760 (long-document late prefill)
  decode256   : steady-state decode near 262k
Each pass records an unprofiled baseline first, then one profiled window.
"""
import json
import statistics
import time
from pathlib import Path

import torch
import torch_npu
from fnx.server import Engine
from profile_support import inventory, make_profiler

OUT = Path(__file__).resolve().parent
SRC = Path('/models/fnx-diagnostics/r141-moe-epilogue-20261111')
TR = OUT / 'traces'
METRIC = 'timing'


def now_ns():
    return time.time_ns()


@torch.no_grad()
def main():
    e = Engine()
    m = e.m
    (OUT / 'inventory.json').write_text(json.dumps(
        dict(stage0=inventory(m.stage0), stage1=m._rpc('inventory')), indent=2))
    tok = e.tok
    pay = json.loads((SRC / 'baseline' / 'payload1.json').read_text())
    text8k = tok.apply_chat_template(pay['messages'], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    ids8k = tok(text8k).input_ids
    assert len(ids8k) == 8064, len(ids8k)
    doc = Path('/root/ascend-dev/eval_text.txt').read_text(encoding='utf-8')
    unit = tok(doc + '\n').input_ids
    target = 262016
    ids256 = (list(unit) * (target // len(unit) + 2))[:target]
    assert len(ids256) == target
    print('PROMPTS_READY', len(ids8k), len(ids256), flush=True)

    runs = []
    state = {'rows': []}

    def csize(remaining):
        choices = [s for s in getattr(m.stage0, 'chunk_graphs', {}) if s <= remaining and s <= m.prefill_limit]
        size = max(choices) if choices else min(remaining, m.maximum, m.prefill_limit)
        if m.stage0._host_pos < 2048 and size > 512:
            size = min(512, remaining)
        return size

    def start(tag):
        m._rpc('prof_start', str(TR / f'{tag}_stage1'), METRIC)
        p = make_profiler(str(TR / f'{tag}_stage0'), METRIC)
        p.start()
        return p

    def stop(p):
        p.stop()
        m._rpc('prof_stop')

    def make_hook(tag):
        box = {}

        def hook(phase, off, size):
            if phase == 'start':
                box['p'] = start(tag)
                box['t0'] = now_ns()
                box['w0'] = time.perf_counter_ns()
            else:
                box['w1'] = time.perf_counter_ns()
                box['t1'] = now_ns()
                stop(box['p'])
                runs.append(dict(tag=tag, kind='prefill_chunk', offset=off, size=size,
                                 window=[box['t0'], box['t1']],
                                 wall_ms=(box['w1'] - box['w0']) / 1e6))
                print('PREFILL_CHUNK_DONE', tag, off, size,
                      round((box['w1'] - box['w0']) / 1e6, 1), flush=True)
        return hook

    def prefill(ids, hooks=None):
        hooks = hooks or {}
        off = 0
        sizes = []
        while off < len(ids):
            size = csize(len(ids) - off)
            sizes.append(size)
            h = hooks.get(off)
            if h:
                h('start', off, size)
            m.prefill_chunk(ids[off:off + size])
            if h:
                m.finish_prefill()
                h('stop', off, size)
            off += size
        m.finish_prefill()
        return sizes

    def dec(tok0, n, prof=None):
        rows = []
        t = int(tok0)
        for _ in range(n):
            s_pc = time.perf_counter_ns()
            s_rt = now_ns()
            if prof is not None:
                with torch.profiler.record_function('r142_step'):
                    m.step_pad(t, True)
            else:
                m.step_pad(t, True)
            e_rt = now_ns()
            t = int(m.last_token)
            rows.append(dict(step=len(rows), start_ns=s_pc, end_ns=time.perf_counter_ns(),
                             realtime_start_ns=s_rt, realtime_end_ns=e_rt))
        return t, rows

    def periods(rows):
        return [(b['end_ns'] - a['end_ns']) / 1e6 for a, b in zip(rows, rows[1:])]

    def record_decode(tag, rows, t0, t1):
        p = periods(rows)
        runs.append(dict(tag=tag, kind='decode_window', rows=rows, window=[t0, t1],
                         periods_ms=p, mean_period_ms=statistics.mean(p),
                         median_period_ms=statistics.median(p)))
        print('DECODE_DONE', tag, round(statistics.mean(p), 3), round(statistics.median(p), 3), len(p), flush=True)

    # ---------------- pass 1: 8k ----------------
    m.reset()
    s8 = prefill(ids8k[:-1], {4096: make_hook('prefill8k')})
    print('PREFILL_8K_SIZES', s8, flush=True)
    t, rows = dec(ids8k[-1], 24)
    t, rows = dec(t, 40)
    record_decode('baseline8k', rows, rows[0]['realtime_start_ns'], rows[-1]['realtime_end_ns'])
    p = start('decode8k')
    t0 = now_ns()
    t, rows = dec(t, 32, prof=p)
    t1 = now_ns()
    stop(p)
    record_decode('decode8k', rows, t0, t1)

    # ---------------- pass 2: 256k prefill ----------------
    m.reset()
    s256 = prefill(ids256[:-1], {0: make_hook('prefill0'), 245760: make_hook('prefillLate')})
    print('PREFILL_256K_CHUNKS', len(s256), s256[:3], '...', s256[-7:], flush=True)
    t, rows = dec(ids256[-1], 8)
    t, rows = dec(t, 40)
    record_decode('baseline256', rows, rows[0]['realtime_start_ns'], rows[-1]['realtime_end_ns'])
    p = start('decode256')
    t0 = now_ns()
    t, rows = dec(t, 32, prof=p)
    t1 = now_ns()
    stop(p)
    record_decode('decode256', rows, t0, t1)

    (OUT / 'runs.json').write_text(json.dumps(runs, indent=2))
    print('R142_PROFILE_COMPLETE', flush=True)
    m.close()


if __name__ == '__main__':
    main()
