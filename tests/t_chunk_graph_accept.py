"""Accept/reject chunk graph using real text, state continuation and wall time."""
import os
import time
import torch
from transformers import AutoTokenizer
from fnx import engine as E
from fnx.model import FnModel


def main():
    n = int(os.environ.get('ACCEPT_N', '64'))
    ctx = int(os.environ.get('FNX_CTX', '2048'))
    m = FnModel(P=ctx, lpc=45, dev0="npu:0", dev1="npu:1", nlayers=48,
                with_head=True, with_ngram=False, verbose=False)
    m.capture()
    tk = AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next')
    text = open('/root/ascend-dev/eval_text.txt', encoding='utf-8').read()
    ids = tk((text + '\n') * (n // 100 + 2)).input_ids[:n+17]
    def run(chunk, offset=0):
        m.reset()
        if offset:
            m.pos0.fill_(offset)
            m.pos1.fill_(offset)
            m._host_pos = offset
        torch.npu.synchronize()
        start = time.perf_counter()
        pre = ids[:n]
        if chunk:
            m.prefill_chunk(pre)
        else:
            for token in pre:
                m.step_pad(token, with_head=False)
        torch.npu.synchronize()
        elapsed = time.perf_counter()-start
        lg = []
        for token in ids[n:n+16]:
            m.step_pad(token)
            lg.append(m.row_logits().float().cpu().clone())
        return elapsed, torch.stack(lg)
    ta, ref = run(False)
    print('BASELINE', ta, n/ta, flush=True)
    m.capture_chunk()
    print('CHUNK_CAPTURED', torch.npu.memory_allocated(0)/2**30, flush=True)
    tb, got = run(True)
    # Recheck decode after shared-pool chunk replay.
    if n <= 128:
        tc, ref2 = run(False)
    else:
        ref2 = ref
    d = (ref-got).abs()
    rel = float(d.mean()/ref.abs().mean())
    labels = torch.tensor(ids[n+1:n+17])
    loss = lambda x: torch.nn.functional.cross_entropy(x, labels).item()
    nll_delta = loss(got)-loss(ref)
    print('RESULT', {'n':n, 'base_tok_s':n/ta, 'chunk_tok_s':n/tb, 'speedup':ta/tb,
                     'maxdiff':float(d.max()), 'rel':rel, 'nll_delta':nll_delta,
                     'top1_agreement':float((ref.argmax(-1)==got.argmax(-1)).float().mean()),
                     'decode_recheck_max':float((ref-ref2).abs().max()) if n<=128 else None}, flush=True)
    assert rel <= 1e-3 and abs(nll_delta) <= .05, 'chunk numerical acceptance failed'
    assert torch.equal(ref, ref2), 'shared pool damaged decode'
    assert tb < ta, 'chunk graph is slower'
    if ctx > 2048 and n <= 128:
        _, a = run(False, 2040)
        _, b = run(True, 2040)
        diff = (a-b).abs()
        boundary_rel = float(diff.mean()/a.abs().mean())
        print('BOUNDARY',float(diff.max()),boundary_rel,flush=True)
        assert boundary_rel < 1e-3, 'short/long graph boundary mismatch'
    print('ACCEPT', flush=True)


if __name__ == '__main__':
    with torch.no_grad():
        main()
