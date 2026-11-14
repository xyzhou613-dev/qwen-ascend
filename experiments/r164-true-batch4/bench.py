"""Full-model static B4 benchmark: four independent sequences in each step.

Private offline runner, not a general HTTP scheduler. Shared document prefix
is prefetched once, state is cloned into four rows before distinct suffixes.
Decode uses one B-row model step and B*10 selected-expert groups per layer.
"""
import contextlib,gc,hashlib,json,os,sys,time,traceback
from pathlib import Path
import torch,torch_npu
from transformers import AutoTokenizer
from config import P
from fnx import engine as E
from fnx.model import FnModel
import batch_ops as O
torch.set_num_threads(1);torch.npu.set_device(0)
O.install();B=E.B
assert B in (1,4) and os.environ['FNX_EXPERIMENT_BATCH_ROWS']=='1'
def emit(name,data):
    (P/name).write_text(json.dumps(data,ensure_ascii=False,indent=2));print(name,json.dumps(data,ensure_ascii=False),flush=True)
def sync():
    for d in (0,1):torch.npu.synchronize(d)
def selected_state(m):
    z={}
    for l in m.layers0+m.layers1:
        for name in ('S','conv'):
            if hasattr(l,name):z[f'{l.idx}:{name}']=getattr(l,name).cpu()
        if l.ltype=='full':
            for name in ('k','v'):z[f'{l.idx}:{name}']=getattr(l.kv,name)[l.ki][:,:,:64].cpu()
            z[f'{l.idx}:index_k']=l.index_k[:,:64].cpu()
            z[f'{l.idx}:index_keys']=l.index_keys[:,:16].cpu()
    return z
def fresh_graph(m):
    for seed,pos in ((16403,17),(16407,23)):
        toks=[101+seed+i*17 for i in range(B)]
        graphs=(m.g0,m.g0_short,m.g1,m._g1_list)
        m.g0=m.g0_short=m.g1=m._g1_list=None
        m.reset();m.pos0.fill_(pos);m.pos1.fill_(pos);m._host_pos=pos
        m.step(toks);sync();eager=m.logits.cpu();hidden=m.h1.cpu();expected=selected_state(m)
        m.g0,m.g0_short,m.g1,m._g1_list=graphs
        m.reset();m.pos0.fill_(pos);m.pos1.fill_(pos);m._host_pos=pos
        m.step(toks);sync();got=selected_state(m)
        assert torch.equal(eager,m.logits.cpu()),('FULL_GRAPH_LOGITS',seed)
        assert torch.equal(hidden,m.h1.cpu()),('FULL_GRAPH_HIDDEN',seed)
        assert all(torch.equal(v,got[k]) for k,v in expected.items()),('FULL_GRAPH_STATE',seed)
    m.reset();return dict(seeds=[16403,16407],positions=[17,23],logits_hidden_states_exact=True)
@contextlib.contextmanager
def prefix_row(m):
    """Single shared prefix only; cannot be used for independent prompts."""
    saved=[];stores=[];old_b=E.B
    for kv in (m.kv0,m.kv1):
        stores.append((kv,kv.k,kv.v));kv.k=[t[:1] for t in kv.k];kv.v=[t[:1] for t in kv.v]
    for l in m.layers0+m.layers1:
        attrs={}
        for n in ('S','conv','index_k','index_keys'):
            v=getattr(l,n,None)
            if v is not None:
                attrs[n]=v;setattr(l,n,v[0] if n.startswith('index_') else v[:1])
        saved.append((l,attrs))
    E.B=1;os.environ['FNX_MOE_GROUP_SPLIT']='32'
    try:yield
    finally:
        E.B=old_b;os.environ['FNX_MOE_GROUP_SPLIT']='0'
        for l,attrs in saved:
            for n,v in attrs.items():
                # index_keys may be reassigned by the original lazy cache.
                current=getattr(l,n)
                if n.startswith('index_'):v[0].copy_(current)
                setattr(l,n,v)
                for i in range(1,old_b):v[i].copy_(v[0])
        for kv,k,v in stores:
            kv.k,kv.v=k,v
            for tensor in k+v:
                for i in range(1,old_b):tensor[i].copy_(tensor[0])
        sync()
def prompts(tok,target):
    text='系统记录：请阅读以下共享材料，等待末尾的问题。\n'
    unit='推理服务需要同时处理多个独立会话，记录性能、状态隔离和每个输出的时间。\n'
    unit_ids=tok.encode(unit,add_special_tokens=False)
    ids=(tok.encode(text,add_special_tokens=False)+unit_ids*((target//len(unit_ids))+2))[:target]
    common=tok.decode(ids)
    rows=[]
    for i in range(B):
        question=common+f'\n本轮任务编号{i}：请写一篇至少六百字的中文说明，解释编号{i}对应的独立会话应如何组织推理服务，分十点详细展开。'
        rendered=tok.apply_chat_template([dict(role='user',content=question)],tokenize=False,add_generation_prompt=True,enable_thinking=False)
        s=tok(rendered,return_tensors='pt').input_ids[0].tolist()
        assert isinstance(s,list) and all(isinstance(v,int) for v in s)
        rows.append(s)
    n=0
    for parts in zip(*rows):
        if len(set(parts))!=1:break
        n+=1
    if B==1:n=max(0,len(rows[0])-32)
    assert len(set(map(len,rows)))==1,'UNEQUAL_SUFFIX_LENGTH'
    assert B==1 or len(set(tuple(x) for x in rows))==B
    return rows,n
def grow_context(m,size):
    """Keep completed 8k evidence before attempting the larger allocation."""
    for graph in (m.g0,m.g0_short,m.g1):
        if graph is not None:graph.reset()
    for graph in m._g1_list or []:graph.reset()
    m.g0=m.g0_short=m.g1=None;m._g1_list=None
    for kv in (m.kv0,m.kv1):kv.k.clear();kv.v.clear()
    for l in m.layers0+m.layers1:
        if l.ltype=='full':l.index_k=l.index_keys=None
    gc.collect()
    for d in (0,1):
        with torch.npu.device(d):torch.npu.empty_cache()
    m.P=size
    m.kv0=E.KVStore(size,m.dev0,sum(l.ltype=='full' for l in m.layers0))
    m.kv1=E.KVStore(size,m.dev1,sum(l.ltype=='full' for l in m.layers1))
    for layers,kv in ((m.layers0,m.kv0),(m.layers1,m.kv1)):
        for l in layers:
            if l.ltype=='full':l.kv=kv;O.prepare(l)
    m.reset();m.capture();return fresh_graph(m)
@torch.no_grad()
def run():
    ctx=int(os.environ.get('FNX_CTX','262144'))
    m=FnModel(P=min(ctx,16384),lpc=24,with_ngram=False)
    for l in m.layers0+m.layers1:O.prepare(l)
    m.capture();check=fresh_graph(m)
    with torch.npu.device(1):
        stream=torch.npu.Stream()
        with torch.npu.stream(stream):
            m._head_device();arg=m.logits.argmax(-1);stream.synchronize()
            hg=torch.npu.NPUGraph()
            with torch.npu.graph(hg,stream=stream):
                m._head_device();arg=m.logits.argmax(-1)
            stream.synchronize()
        # Fresh input checks cover the additional head graph separately.
        for seed in (16412,16413):
            torch.manual_seed(seed);m.h1.normal_(0,.02);m._head_device()
            expected=m.logits.cpu();expected_ids=m.logits.argmax(-1).cpu()
            hg.replay();sync()
            assert torch.equal(expected,m.logits.cpu()) and torch.equal(expected_ids,arg.cpu()),'HEAD_STALE'
    emit(f'batch{B}-graph-check.json',check)
    tokenizer=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next',trust_remote_code=True)
    results=[];eos=tokenizer.eos_token_id
    for target in (8000,261900):
        if target+200>ctx:continue
        if target+200>m.P:
            long_check=grow_context(m,ctx)
            emit(f'batch{B}-long-graph-check.json',long_check)
        for rep in range(3):
            rows,n=prompts(tokenizer,target);m.reset();begin=time.perf_counter_ns()
            with prefix_row(m):m.prefill_chunk(rows[0][:n])
            for j in range(n,len(rows[0])):m.step([r[j] for r in rows],with_head=False)
            with torch.npu.device(1):hg.replay();next_ids=arg.cpu().tolist()
            first=time.perf_counter_ns();outputs=[[] for _ in range(B)];alive=[True]*B;events=[]
            hidden_hashes=[hashlib.sha256(row.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for row in m.h1]
            assert len(set(hidden_hashes))==B,'DUPLICATED_ROW_HIDDEN_STATE'
            for step in range(128):
                now=time.perf_counter_ns()
                for i in range(B):
                    if alive[i]:
                        outputs[i].append(int(next_ids[i]));events.append(dict(row=i,step=step,ns=now))
                        if next_ids[i]==eos:alive[i]=False
                if not any(alive) or step==127:break
                m.step(next_ids,with_head=False)
                with torch.npu.device(1):hg.replay();next_ids=arg.cpu().tolist()
            end=time.perf_counter_ns();byrow=[]
            for i,out in enumerate(outputs):
                stamps=[v['ns'] for v in events if v['row']==i]
                duration=(stamps[-1]-stamps[0])/1e9
                byrow.append(dict(row=i,output_tokens=len(out),decode_tps=(len(out)-1)/duration if duration else None,
                    eos=bool(out and out[-1]==eos),text=tokenizer.decode(out)))
            lo=min(x['ns'] for x in events);hi=max(x['ns'] for x in events)
            count=sum(max(0,len(x)-1) for x in outputs)
            duration=(hi-lo)/1e9
            r=dict(batch=B,rep=rep,requested_context=target,input_tokens=len(rows[0]),shared_prefix_tokens=n,
                ttft_s=(first-begin)/1e9,decode_union_s=duration,
                total_decode_tps=count/duration if duration else None,
                end_to_end_output_tps=sum(map(len,outputs))/((end-begin)/1e9),
                per_row=byrow,timeline=events,ops=dict(O.AUDIT),
                input_hashes=[hashlib.sha256(json.dumps(r).encode()).hexdigest() for r in rows],
                prefill_hidden_hashes=hidden_hashes,
                allocated_gib=[torch.npu.memory_allocated(i)/2**30 for i in (0,1)])
            assert B==1 or r['ops']==dict(moe_rows=4,gmm_groups=40,qsa_rows=4),r['ops']
            results.append(r);emit(f'batch{B}-results.json',dict(status='partial',records=results,
                scope='full model static independent-row batch; shared-prefix prefill; unmodified BF16 head'))
    emit(f'batch{B}-results.json',dict(status='completed',graph_check=check,records=results,
        scope='full model static independent-row batch; shared-prefix prefill; unmodified BF16 head'))
    sys.stdout.flush();os._exit(0)
try:run()
except Exception as e:
    emit(f'batch{B}-failure.json',dict(error=repr(e)));traceback.print_exc();sys.stderr.flush();os._exit(1)
