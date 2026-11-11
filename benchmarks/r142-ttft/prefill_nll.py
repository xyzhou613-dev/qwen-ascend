"""Teacher-forced NLL on a real 8k document after the production chunk policy.

The production /fnx/loss endpoint feeds tokens one at a time (decode path) and
cannot see a prefill-chunking change. This harness runs the same 8064-token
payload the HTTP benchmark uses, prefills all but the last 200 tokens through
the real chunk policy, then scores those 200 positions.
"""
import json
import sys
from pathlib import Path

import torch
from fnx.server import Engine

OUT = Path(__file__).resolve().parent
SRC = Path('/models/fnx-diagnostics/r141-moe-epilogue-20261111')


@torch.no_grad()
def main():
    first = sys.argv[1]
    e = Engine()
    m = e.m
    pay = json.loads((SRC / 'baseline' / 'payload1.json').read_text())
    text = e.tok.apply_chat_template(pay['messages'], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=False)
    ids = e.tok(text).input_ids
    L = len(ids)
    pre = L - 200
    m.reset()
    m.plan_prefill(pre)
    m.prefill_chunk(ids[:pre])
    m.finish_prefill()
    torch.npu.synchronize()
    total = 0.0
    n = 0
    hit = 0
    for i in range(pre, L - 1):
        m.step_pad(int(ids[i]), True)
        lg = m.row_logits().float().view(-1)
        lp = torch.log_softmax(lg, dim=-1)
        tgt = int(ids[i + 1])
        total += float(-lp[tgt])
        hit += int(int(lg.argmax()) == tgt)
        n += 1
    print('PREFILL_NLL', json.dumps(dict(first_block=first, tokens=L, scored=n,
                                         mean_nll=round(total / n, 6),
                                         ppl=round(2.718281828459045 ** (total / n), 4),
                                         top1=round(hit / n, 4))), flush=True)
    m.close()


if __name__ == '__main__':
    main()
