"""Bounded real-layer B4 correctness, isolation and graph timing; no Engine."""
import copy,gc,json,os,statistics,sys,time,traceback,urllib.request
import torch,torch_npu
from config import P
from fnx import engine as E
from fnx.load import load_layer,QDIR
import batch_ops as O
d=int(sys.argv[1]);started=time.monotonic();baseline=None
torch.set_num_threads(1);torch.npu.set_device(0)
torch.npu.set_per_process_memory_fraction(1/24,0)
assert E.B==4 and E.__file__==str(P/'runtime/fnx/engine.py')
O.install()
def guard():
    global baseline
    h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
    assert h['status']=='ok' and h['service_revision']=='r152' and h['worker_alive']
    if baseline is None:baseline=h['pipeline_transfers']
    assert baseline==h['pipeline_transfers'],'BUSINESS_PROGRESS_ABORT'
    assert torch.npu.mem_get_info()[0]>2*1024**3,'PHYSICAL_RESERVE'
    assert time.monotonic()-started<370,'TIME_LIMIT'
def metrics(a,b):
    a=a.cpu().float();b=b.cpu().float()
    return dict(relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),
                max_abs=float((a-b).abs().max()),exact=bool(torch.equal(a,b)),finite=bool(torch.isfinite(b).all()))
def state(l):
    z={'conv':l.conv.cpu()}
    if hasattr(l,'S'):z['S']=l.S.cpu()
    else:
        z.update(k=l.kv.k[0].cpu(),v=l.kv.v[0].cpu(),index_k=l.index_k.cpu(),index_keys=l.index_keys.cpu())
    return z
def load(l,z):
    for k,v in z.items():
        target=l.kv.k[0] if k=='k' else l.kv.v[0] if k=='v' else getattr(l,k)
        target.copy_(v)
def row(l,z,i,legacy=False):
    r=copy.copy(l);r.conv=z['conv'][i:i+1].to(l.dev)
    if 'S' in z:r.S=z['S'][i:i+1].to(l.dev)
    else:
        r.kv=copy.copy(l.kv);r.kv.k=[z['k'][i:i+1].to(l.dev)];r.kv.v=[z['v'][i:i+1].to(l.dev)]
        r.index_k=z['index_k'][i:i+1].to(l.dev);r.index_keys=z['index_keys'][i:i+1].to(l.dev)
        if legacy:r.index_k=r.index_k[0];r.index_keys=r.index_keys[0]
    return r
@torch.no_grad()
def run():
    records=[];guard()
    for idx in (24*d,24*d+3):
        guard();E._GAMMA_CACHE.clear();torch.manual_seed(16400+idx)
        w=load_layer(f'{QDIR}/L{idx:02d}.safetensors',idx,'npu:0')
        l=E.FnLayer(idx,'linear' if idx%4!=3 else 'full',w,'npu:0');del w
        if l.ltype=='full':
            l.kv=E.KVStore(8192,'npu:0',1);O.prepare(l)
            l.kv.k[0].normal_(0,.02);l.kv.v[0].normal_(0,.02);l.index_k.normal_(0,.03)
            raw=l.index_k.reshape(4,2048,4,128).float().mean(2).to(torch.bfloat16)
            cs,sn=E.rope_cs(torch.arange(2048,device='npu')*4,l.inv)
            l.index_keys.copy_(E.apply_rope(E.rms_norm(raw,l.index_kn),cs[None,:,:],sn[None,:,:]).float())
        else:l.S.normal_(0,.003)
        l.conv.normal_(0,.003)
        h=torch.randn(4,1,E.HCH,device='npu',dtype=torch.bfloat16)*.03
        pos=torch.tensor([3,511,2047,4093],device='npu',dtype=torch.int64)
        initial=state(l);out=l.forward(h,pos).clone();after=state(l)
        independent=[];legacy=[]
        for i in range(4):
            guard();r=row(l,initial,i);independent.append(r.forward(h[i:i+1],pos[i:i+1]).clone());del r
            r=row(l,initial,i,legacy=True)
            # Bind original B1 decode to the shallow object sharing weights.
            r._moe=O.ORIGINAL_MOE.__get__(r);r._qsa=O.ORIGINAL_QSA.__get__(r)
            legacy.append(r.forward(h[i:i+1],pos[i:i+1]).clone());del r
        candidate=metrics(torch.cat(independent),out);reference=metrics(torch.cat(legacy),out)
        assert candidate['finite'] and candidate['relative_l2']<.01,('BATCH_ROW_EQUIVALENCE',candidate)
        load(l,initial);changed=h.clone();changed[0]*=2
        changed_out=l.forward(changed,pos).clone();changed_state=state(l)
        assert torch.equal(out[1:],changed_out[1:]),'CROSS_ROW_OUTPUT_LEAK'
        assert all(torch.equal(after[k][1:],changed_state[k][1:]) for k in after),'CROSS_ROW_STATE_LEAK'
        perm=torch.tensor([2,0,3,1],device='npu')
        load(l,{k:v.index_select(0,perm.cpu()) for k,v in initial.items()})
        perm_out=l.forward(h.index_select(0,perm),pos.index_select(0,perm)).clone()
        permutation=metrics(out.index_select(0,perm),perm_out)
        assert permutation['relative_l2']<.01
        load(l,initial);stream=torch.npu.Stream()
        with torch.npu.stream(stream):
            l.forward(h,pos);stream.synchronize()
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph,stream=stream):captured=l.forward(h,pos)
            stream.synchronize();fresh=[]
            for seed in (16491,16492):
                guard();torch.manual_seed(seed);h.normal_(0,.04);load(l,initial)
                eager=l.forward(h,pos).clone();expected=state(l)
                load(l,initial);graph.replay();stream.synchronize()
                assert torch.equal(captured,eager),'STALE_GRAPH_OUTPUT'
                got=state(l)
                assert all(torch.equal(got[k],expected[k]) for k in expected),'STALE_GRAPH_STATE'
                fresh.append(seed)
            times=[]
            for _ in range(7):
                guard();a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True)
                a.record(stream)
                for _ in range(8):graph.replay()
                b.record(stream);b.synchronize();times.append(a.elapsed_time(b)/8)
            graph.reset()
        r=row(l,initial,0);one=h[:1].clone();one_pos=pos[:1].clone()
        with torch.npu.stream(stream):
            r.forward(one,one_pos);stream.synchronize();g1=torch.npu.NPUGraph()
            with torch.npu.graph(g1,stream=stream):one_out=r.forward(one,one_pos)
            stream.synchronize();one_times=[]
            for _ in range(7):
                guard();a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record(stream)
                for _ in range(8):g1.replay()
                b.record(stream);b.synchronize();one_times.append(a.elapsed_time(b)/8)
            g1.reset()
        records.append(dict(layer=idx,kind=l.ltype,batch_rows=4,selected_gmm_groups=40,
            batch_vs_independent=candidate,batch_vs_original_b1=reference,
            other_rows_output_state_exact=True,permutation=permutation,fresh_graph_seeds=fresh,
            batch4_layer_median_ms=statistics.median(times),samples_ms=times,
            matched_batch1_layer_median_ms=statistics.median(one_times),
            allocated_gib=torch.npu.memory_allocated()/2**30))
        print(json.dumps(records[-1]),flush=True)
        del l,r,g1,one,one_pos,one_out,initial,out,after,independent,legacy,changed_out,changed_state,changed,perm_out,h,captured,graph,eager,expected,got
        gc.collect();torch.npu.empty_cache()
    guard();return dict(status='passed',device=d,elapsed_s=time.monotonic()-started,records=records,
                        production_transfers_unchanged=baseline,scope='real layer only; not full-model throughput')
try:
    result=run();(P/f'probe-device{d}.json').write_text(json.dumps(result,indent=2));sys.stdout.flush();os._exit(0)
except Exception as ex:
    (P/f'probe-device{d}.json').write_text(json.dumps(dict(status='failed',error=repr(ex)),indent=2))
    traceback.print_exc();sys.stderr.flush();os._exit(1)
