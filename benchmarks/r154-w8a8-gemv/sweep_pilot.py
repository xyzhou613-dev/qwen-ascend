"""Layer-at-a-time, teacher-forced NLL pilot; never constructs an Engine.

Must establish agreement with the live r152 /fnx/loss baseline before this
method can support broader quality claims. Only one real layer is on the NPU.
"""
import gc,json,os,time,urllib.request
from pathlib import Path
os.environ['FNX_CTX']='8192'
import torch,torch_npu
from safetensors import safe_open
from transformers import AutoTokenizer
from fnx import engine as E
from fnx.load import load_layer,load_nonlayer,QDIR,SRCDIR
from fnx.model import FnModel
from fnx.w8a8 import configure_layer
P=Path(__file__).resolve().parent;started=time.monotonic()
torch.set_num_threads(1);torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(1/32,0)
TEXT='A reliable inference service needs reproducible measurements. We compare the same inputs, record latency and accuracy, and check whether the result survives repeated tests.'

def guard():
    h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
    assert h['service_revision']=='r152' and h['worker_alive']
    assert torch.npu.mem_get_info()[0]>512*1024**2
    assert time.monotonic()-started<1500

def reset(L):
    for name in ('S','conv','ple_conv','index_k','index_keys'):
        tensor=getattr(L,name,None)
        if tensor is not None:tensor.zero_()
    if L.kv:
        for t in L.kv.k+L.kv.v:t.zero_()

@torch.no_grad()
def run_sequence(L,hidden):
    x=torch.zeros(1,1,E.HCH,device='npu',dtype=torch.bfloat16)
    pos=torch.zeros(1,device='npu',dtype=torch.int64);stream=torch.npu.Stream()
    with torch.npu.stream(stream):
        reset(L);L.forward(x,pos);stream.synchronize();g=torch.npu.NPUGraph()
        with torch.npu.graph(g,stream=stream):out=L.forward(x,pos)
        stream.synchronize();reset(L);stream.synchronize()
        result=[]
        for t,row in enumerate(hidden):
            guard();x.copy_(row.view(1,1,E.HCH));pos.fill_(t)
            g.replay();stream.synchronize();result.append(out.detach().cpu())
        g.reset()
    return torch.cat(result,0)

@torch.no_grad()
def head_scores(hidden,ids,mixer):
    mn,md,mu=mixer
    mixed=[]
    for row in hidden:
        h=row.view(1,1,E.HCH).npu()
        mixed.append(E.gated_residual(h,mn,md,mu,None).view(1,E.H).cpu())
    mixed=torch.cat(mixed).npu();logits=torch.empty(len(ids)-1,E.VOCAB,dtype=torch.bfloat16)
    with safe_open(str(Path(QDIR)/'nonlayer.safetensors'),framework='pt',device='cpu') as f:
        for first in range(0,E.VOCAB,62080):
            last=min(first+62080,E.VOCAB)
            q=f.get_slice('lm_head.weight.q8')[first:last].npu()
            s=f.get_slice('lm_head.weight.s8')[first:last].npu()
            obj=E.DenseW8(q,s,K=E.H)
            # Match production's four chunked W8A16 calls, row by row.
            logits[:,first:last].copy_(torch.cat([obj(h[None]) for h in mixed]).cpu())
            del obj,q,s;torch.npu.empty_cache()
    lp=logits.float().log_softmax(-1);target=torch.tensor(ids[1:])
    losses=-lp[torch.arange(len(target)),target]
    return dict(mean_nll=float(losses.mean()),nll_tokens=losses.tolist(),
                argmax=logits.argmax(-1).tolist(),top1=float((logits.argmax(-1)==target).float().mean()))

@torch.no_grad()
def main():
    guard();tok=AutoTokenizer.from_pretrained(SRCDIR);ids=tok.encode(TEXT)[:17]
    assert len(ids)==17
    emb,unused,mx=load_nonlayer('npu:0','npu:0',with_lmhead=False)
    dummy=FnModel.__new__(FnModel);dummy.emb_q,dummy.emb_s=emb
    initial=dummy.embed_rows(ids[:-1]).view(16,1,E.H).repeat(1,1,E.HC)
    del dummy,emb,unused
    paths={mode:initial.clone() for mode in ('baseline','candidate')}
    cfg=json.load(open(Path(SRCDIR)/'config.json'))['text_config'];records=[]
    mixer=tuple(mx[n] for n in ('hc_norm.weight','input_mix_weight_down.weight','input_mix_weight_up.weight'))
    for index in range(48):
        guard();os.environ['FNX_DENSE_W8A8']='0';E._GAMMA_CACHE.clear()
        ltype='full' if cfg['layer_types'][index] in E.FULL_TYPES else 'linear'
        W=load_layer(f'{QDIR}/L{index:02d}.safetensors',index,'npu:0')
        L=E.FnLayer(index,ltype,W,'npu:0');del W
        if ltype=='full':
            L.kv=E.KVStore(8192,'npu:0',1);L.short_qsa=True
            L.index_k=torch.zeros(8192,128,device='npu',dtype=torch.bfloat16)
            L.index_keys=torch.zeros(2048,128,device='npu',dtype=torch.float32)
        torch.npu.synchronize();torch.npu.empty_cache()
        paths['baseline']=run_sequence(L,paths['baseline'])
        replaced=configure_layer(L,E.Raw)
        assert replaced,('unconverted layer',index)
        torch.npu.synchronize();torch.npu.empty_cache()
        paths['candidate']=run_sequence(L,paths['candidate'])
        a=paths['baseline'].float();b=paths['candidate'].float()
        records.append(dict(layer=index,replacements=replaced,hidden_relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12))))
        print('SWEEP_LAYER',index,records[-1]['hidden_relative_l2'],flush=True)
        del L;E._GAMMA_CACHE.clear();gc.collect();torch.npu.empty_cache()
        (P/'sweep-pilot-progress.json').write_text(json.dumps(dict(completed_layers=index+1,records=records),indent=2))
    scores={mode:head_scores(hidden,ids,mixer) for mode,hidden in paths.items()}
    (P/'sweep-pilot.json').write_text(json.dumps(dict(text=TEXT,ids=ids,records=records,scores=scores,
        scope='teacher-forced decode only; 16 targets; not full model task-quality acceptance',
        status='requires_live_baseline_agreement',elapsed_s=time.monotonic()-started),indent=2))
    print('SWEEP_PILOT_DONE',json.dumps({k:v['mean_nll'] for k,v in scores.items()}),flush=True)

if __name__=='__main__':main()
