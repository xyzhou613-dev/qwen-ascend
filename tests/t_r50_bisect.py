"""Compare each candidate layer on real baseline activations and identical state."""
import os
import torch
from transformers import AutoTokenizer
from fnx import engine as E
from fnx.model import FnModel


def mode(which):
    if which=='base':
        os.environ['FNX_MOE_GROUP']='0';os.environ['FNX_PREFILL_DN']='torch';os.environ['FNX_QSA_GQA']='0'
    else:
        os.environ['FNX_MOE_GROUP']='1';os.environ['FNX_PREFILL_DN']='fp32_chunk';os.environ['FNX_QSA_GQA']='1'


def main():
    os.environ['FNX_MOE_GROUP']='1'
    m=FnModel(P=8192,lpc=45,dev0='npu:0',dev1='npu:1',nlayers=48,with_head=False,with_ngram=False,verbose=False)
    tk=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next')
    ids=tk('昇腾处理器面向大规模深度学习训练与推理场景。').input_ids[:8]
    T=len(ids)
    h=m.embed_rows(ids).to(m.dev0).view(T,1,E.H).repeat(1,1,E.HC)
    pos=torch.tensor([0],dtype=torch.int64,device=m.dev0)
    for L in m.layers0+m.layers1:
        if str(h.device)!=str(torch.device(L.dev)):
            h=h.cpu().to(L.dev);pos=pos.cpu().to(L.dev)
        with torch.npu.device(L.dev):
            L.short_qsa=True
            state={'conv':L.conv.clone()}
            if hasattr(L,'S'):state['S']=L.S.clone()
            if L.kv is not None:
                state['k']=L.kv.k[L.ki].clone();state['v']=L.kv.v[L.ki].clone();state['ik']=L.index_k.clone()
            def restore():
                L.conv.copy_(state['conv'])
                if 'S'in state:L.S.copy_(state['S'])
                if 'k'in state:
                    L.kv.k[L.ki].copy_(state['k']);L.kv.v[L.ki].copy_(state['v']);L.index_k.copy_(state['ik'])
            mode('base');ref=L.forward_chunk(h,pos,T).clone()
            restore();mode('candidate');got=L.forward_chunk(h,pos,T)
            d=(ref.float()-got.float()).abs()
            rel=float(d.mean()/(ref.float().abs().mean()+1e-9))
            print('LAYER_DIFF',L.idx,rel,float(d.max()),flush=True)
            if rel>.01:
                for component in ('dn','moe','qsa'):
                    restore();mode('base')
                    if component=='dn':os.environ['FNX_PREFILL_DN']='fp32_chunk'
                    if component=='moe':os.environ['FNX_MOE_GROUP']='1'
                    if component=='qsa':os.environ['FNX_QSA_GQA']='1'
                    got=L.forward_chunk(h,pos,T)
                    diff=(ref.float()-got.float()).abs()
                    print('COMPONENT',L.idx,component,float(diff.mean()/(ref.float().abs().mean()+1e-9)),float(diff.max()),flush=True)
                break
            h=ref


if __name__=='__main__':
    with torch.no_grad():main()
