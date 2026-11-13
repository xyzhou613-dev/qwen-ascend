"""Paired teacher-forced decode NLL with one loaded real layer at a time."""
import gc,json,os,time
from pathlib import Path
import sweep_pilot as S
import torch,torch_npu
from fnx import engine as E
from fnx.load import load_layer,load_nonlayer,QDIR,SRCDIR
from fnx.model import FnModel
from smooth_w8a8 import configure_layer
P=Path(__file__).resolve().parent;started=time.monotonic()

def guard():
    # The pilot's safety checks are retained, with a bounded suite time budget.
    S.started=time.monotonic()
    S.guard()
    assert time.monotonic()-started<2300

@torch.no_grad()
def run_documents(L,documents):
    x=torch.zeros(1,1,E.HCH,device='npu',dtype=torch.bfloat16)
    pos=torch.zeros(1,device='npu',dtype=torch.int64);stream=torch.npu.Stream()
    with torch.npu.stream(stream):
        S.reset(L);L.forward(x,pos);stream.synchronize();g=torch.npu.NPUGraph()
        with torch.npu.graph(g,stream=stream):out=L.forward(x,pos)
        stream.synchronize();result=[]
        for hidden in documents:
            guard();S.reset(L);stream.synchronize();rows=[]
            for t,row in enumerate(hidden):
                if t%8==0:guard()
                x.copy_(row.view(1,1,E.HCH));pos.fill_(t);g.replay();stream.synchronize()
                rows.append(out.detach().cpu())
            result.append(torch.cat(rows,0))
        g.reset()
    return result

@torch.no_grad()
def main():
    guard();cases=json.load(open(P/'suite-inputs.json'))['cases']
    emb,unused,mx=load_nonlayer('npu:0','npu:0',with_lmhead=False)
    dummy=FnModel.__new__(FnModel);dummy.emb_q,dummy.emb_s=emb
    initial=[dummy.embed_rows(case['ids'][:-1]).view(-1,1,E.H).repeat(1,1,E.HC) for case in cases]
    del dummy,emb,unused
    paths={mode:[h.clone() for h in initial] for mode in ('baseline','candidate')}
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
        paths['baseline']=run_documents(L,paths['baseline'])
        replacements=configure_layer(L,E.Raw);assert replacements
        torch.npu.synchronize();torch.npu.empty_cache()
        paths['candidate']=run_documents(L,paths['candidate'])
        records.append(dict(layer=index,replacements=replacements))
        print('SUITE_LAYER',index,'elapsed',round(time.monotonic()-started,1),flush=True)
        del L;E._GAMMA_CACHE.clear();gc.collect();torch.npu.empty_cache()
        (P/'sweep-suite-smooth-progress.json').write_text(json.dumps(dict(completed_layers=index+1,records=records),indent=2))
    scores=[]
    for i,case in enumerate(cases):
        guard();row=dict(name=case['name'],scored_tail=case['scored_tail'])
        for mode in paths:
            value=S.head_scores(paths[mode][i],case['ids'],mixer)
            tail=value['nll_tokens'][-case['scored_tail']:]
            value['tail_mean_nll']=sum(tail)/len(tail);row[mode]=value
        row['tail_delta_nll']=row['candidate']['tail_mean_nll']-row['baseline']['tail_mean_nll']
        scores.append(row);print('SUITE_SCORE',row['name'],row['tail_delta_nll'],flush=True)
        (P/'sweep-suite-smooth-scores.json').write_text(json.dumps(scores,indent=2))
    summary=dict(status='passed_execution',quality_scope='development teacher-forced decode; not task accuracy or prefill acceptance',
        targets=sum(r['scored_tail'] for r in scores),cases=len(scores),
        mean_delta_nll=sum(r['tail_delta_nll']*r['scored_tail'] for r in scores)/sum(r['scored_tail'] for r in scores),
        worst_case_delta_nll=max(r['tail_delta_nll'] for r in scores),elapsed_s=time.monotonic()-started,
        scores=scores,exclusive=False,max_allocated_bytes=torch.npu.max_memory_allocated())
    (P/'sweep-suite-smooth.json').write_text(json.dumps(summary,indent=2));print('SUITE_DONE',summary['mean_delta_nll'],summary['worst_case_delta_nll'],flush=True)

if __name__=='__main__':main()
