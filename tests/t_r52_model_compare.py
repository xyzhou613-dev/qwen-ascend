"""End-to-end sparse versus original dense masked attention at 8k capacity."""
import os,time
import torch
from transformers import AutoTokenizer
from fnx.model import FnModel

def main():
    os.environ['FNX_QSA_SPARSE']='0'
    m=FnModel(P=8192,lpc=40,dev0='npu:0',dev1='npu:1',nlayers=48,with_head=True,with_ngram=False,verbose=False)
    tk=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next')
    ids=tk((open('/root/ascend-dev/eval_text.txt',encoding='utf-8').read()+'\n')*20).input_ids
    def run(chunk,n,offset):
        m.reset();m.pos0.fill_(offset);m.pos1.fill_(offset);m._host_pos=offset
        if chunk:m.prefill_chunk(ids[:n])
        else:
            for token in ids[:n]:m.step_pad(token,with_head=False)
        rows=[]
        for token in ids[n:n+64]:
            m.step_pad(token);rows.append(m.row_logits().float().cpu().clone())
        return torch.stack(rows)
    m.capture()
    ref=run(False,512,2040)
    os.environ['FNX_QSA_SPARSE']='1'
    m.capture();m.capture_chunk()
    got=run(True,512,2040)
    labels=torch.tensor(ids[513:577])
    loss=lambda x:float(torch.nn.functional.cross_entropy(x,labels))
    kl=float(torch.nn.functional.kl_div(got.log_softmax(-1),ref.log_softmax(-1),log_target=True,reduction='batchmean'))
    agreement=float((ref.argmax(-1)==got.argmax(-1)).float().mean())
    print('MODEL_COMPARE',{'ref_nll':loss(ref),'sparse_nll':loss(got),'delta':loss(got)-loss(ref),'kl':kl,'agreement':agreement},flush=True)
    assert abs(loss(got)-loss(ref))<.05 and kl<.02 and agreement>=.95
    host=m.save_state(to_host=True)
    before=m.state_to_host(m.save_state())
    def equal(a,b):
        if torch.is_tensor(a):return torch.equal(a,b)
        if isinstance(a,dict):return all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return all(equal(x,y) for x,y in zip(a,b))
        return a==b
    assert equal(host,before)
    m.reset();m.load_state(host)
    assert equal(m.save_state(to_host=True),before)
    print('MODEL_COMPARE_ACCEPT snapshot_exact=True',flush=True)

if __name__=='__main__':
    with torch.no_grad():main()
