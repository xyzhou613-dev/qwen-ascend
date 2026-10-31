"""Full model: compare grouped parallel prefill against unchanged decode."""
import os,time
import torch
from transformers import AutoTokenizer
from fnx import engine as E
from fnx.model import FnModel


def main():
    m=FnModel(P=8192,lpc=int(os.environ.get('FNX_LPC','45')),dev0='npu:0',dev1='npu:1',
              nlayers=48,with_head=True,with_ngram=False,verbose=False)
    m.capture()
    tk=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next')
    text=open('/root/ascend-dev/eval_text.txt',encoding='utf-8').read()
    ids=tk((text+'\n')*40).input_ids
    def run(chunk,n,offset=0):
        m.reset()
        if offset:
            m.pos0.fill_(offset); m.pos1.fill_(offset); m._host_pos=offset
        torch.npu.synchronize(); start=time.perf_counter()
        if chunk:m.prefill_chunk(ids[:n])
        else:
            for token in ids[:n]:m.step_pad(token,with_head=False)
        torch.npu.synchronize(); elapsed=time.perf_counter()-start
        rows=[]
        for token in ids[n:n+128]:
            m.step_pad(token)
            rows.append(m.row_logits().float().cpu().clone())
        return torch.stack(rows),elapsed
    n=int(os.environ.get('ACCEPT_N','256'))
    ref,ta=run(False,n)
    print('REFERENCE',n,ta,n/ta,flush=True)
    if not int(os.environ.get('FNX_MOE_GROUP_SPLIT','0')):
        m.capture_chunk()
    print('CAPTURED',E._CHUNK,torch.npu.memory_allocated(0)/2**30,flush=True)
    got,tb=run(True,n)
    def compare(name,a,b,n):
        labels=torch.tensor(ids[n+1:n+129])
        loss=lambda z:float(torch.nn.functional.cross_entropy(z,labels))
        d=(a-b).abs()
        metrics={'maxdiff':float(d.max()),'rel':float(d.mean()/a.abs().mean()),
                 'ref_nll':loss(a),'got_nll':loss(b),'nll_delta':loss(b)-loss(a),
                 'top1_agreement':float((a.argmax(-1)==b.argmax(-1)).float().mean()),
                 'kl':float(torch.nn.functional.kl_div(b.log_softmax(-1),a.log_softmax(-1),log_target=True,reduction='batchmean'))}
        print(name,metrics,flush=True)
        assert abs(metrics['nll_delta'])<=.05, 'NLL acceptance failed'
        assert metrics['top1_agreement']>=.95, 'top1 agreement acceptance failed'
    compare('QUALITY',ref,got,n)
    print('SPEED',{'tokens':n,'ref_tok_s':n/ta,'chunk_tok_s':n/tb,'speedup':ta/tb},flush=True)
    for chunk in ((256,512) if int(os.environ.get('FNX_MOE_GROUP_SPLIT','0')) else ()):
        E._CHUNK=chunk
        candidate,elapsed=run(True,n)
        compare('SWEEP_QUALITY_'+str(chunk),ref,candidate,n)
        print('SWEEP_SPEED',chunk,n/elapsed,flush=True)
    boundary_n=128 if int(os.environ.get('FNX_MOE_GROUP_SPLIT','0')) else E._CHUNK
    ref,_=run(False,boundary_n,2040)
    got,_=run(True,boundary_n,2040)
    compare('BOUNDARY',ref,got,boundary_n)
    assert tb<ta
    print('MODEL_ACCEPT',flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
