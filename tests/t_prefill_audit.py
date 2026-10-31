"""Compare full-model prefill states and logits, with no production mutation."""
import os
import time
import torch
from fnx import engine as E
from fnx.model import FnModel


def main():
    m = FnModel(P=2048, lpc=45, dev0="npu:0", dev1="npu:1", nlayers=48,
                with_head=True, with_ngram=False, verbose=False)
    ids = [18643, 11111, 22222, 33333, 44444, 55555, 66666, 77777]
    def run(ch):
        m.reset()
        E._CHUNK = ch
        t = time.perf_counter()
        if ch == 0:
            for x in ids:
                m.step_pad(x, with_head=False)
        else:
            m.prefill_chunk(ids)
        torch.npu.synchronize()
        print("PREFILL", ch, (time.perf_counter()-t)*1000/len(ids), "ms/token", flush=True)
        state = m.save_state()
        m.step_pad(12345)
        return state, m.row_logits().float().cpu().clone()
    ref, logits = run(0)
    for ch in (1, 8):
        st, lg = run(ch)
        for i, (a, b) in enumerate(zip(ref['layers'], st['layers'])):
            for key in a:
                if torch.is_tensor(a[key]):
                    d = (a[key].float()-b[key].float()).abs()
                    rel = float(d.mean()/(a[key].float().abs().mean()+1e-9))
                    if rel > 1e-6:
                        print('STATE_DIFF', ch, i, key, float(d.max()), rel, flush=True)
        d = (logits-lg).abs()
        print('LOGITS', ch, 'max', float(d.max()), 'rel', float(d.mean()/logits.abs().mean()),
              'argmax', int(logits.argmax()), int(lg.argmax()), flush=True)


if __name__ == '__main__':
    with torch.no_grad():
        main()
