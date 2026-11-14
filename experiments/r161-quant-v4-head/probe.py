"""Paired BF16/W8A8 V1/V2/V4 layers plus speculative state lifecycle checks.

Synthetic hidden states, 8k zero KV context. No Engine or drafter is built.
CPU full-cache fixtures are validation-only; timed backup is O(T) rows.
"""
import gc,hashlib,json,os,statistics,sys,time,urllib.request
from pathlib import Path
import torch,torch_npu
from fnx import engine as E
from fnx.load import load_layer,QDIR
from fnx.smooth_w8a8 import SmoothW8A8
import small_delta
from verifier_state import PrefixState,capture_cpu,load_cpu,equal_cpu,expected_prefix,conv_journals

P=Path(__file__).resolve().parent;device=int(sys.argv[1]);started=time.monotonic();transfers=None
torch.set_num_threads(1);torch.npu.set_device(0)
# Two layer objects share original quantized tensors; journal/capture scratch is bounded.
torch.npu.set_per_process_memory_fraction(1/24,0)
assert E.__file__==str(P/'runtime/fnx/engine.py')
assert os.environ['FNX_MOE_GROUP']=='0' and os.environ['FNX_KMAJOR_DOWN']=='0'
call_hist={};object_names={};old_call=SmoothW8A8.__call__
def observed_call(self,x):
    key=object_names.get(id(self),'unregistered')
    row=call_hist.setdefault(key,{})
    n=str(x.numel()//self.K);row[n]=row.get(n,0)+1
    return old_call(self,x)
SmoothW8A8.__call__=observed_call

def guard():
    global transfers
    h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
    assert h['status']=='ok' and h['service_revision']=='r152' and h['worker_alive']
    if transfers is None:transfers=h['pipeline_transfers']
    assert h['pipeline_transfers']==transfers,'PRODUCTION_REQUEST_PROGRESS_ABORT_PROBE'
    assert torch.npu.mem_get_info()[0]>2*1024**3,'PHYSICAL_MEMORY_RESERVE'
    assert time.monotonic()-started<370,'PROBE_TIME_LIMIT'

def err(a,b):
    a=a.cpu().float();b=b.cpu().float()
    return dict(relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),
        max_abs=float((a-b).abs().max()),finite=bool(torch.isfinite(b).all()))

def seed_state(layer,seed):
    torch.manual_seed(seed);layer.conv.copy_(torch.randn_like(layer.conv)*.003)
    if layer.ltype in ('linear','ple'):layer.S.copy_(torch.randn_like(layer.S)*.01)
    else:
        for x in layer.kv.k+layer.kv.v:x.zero_()
        layer.index_k.zero_();layer.index_keys.zero_()

def verify_call(layer,inputs,pos,T,backup):
    if T==1:return layer.forward(inputs,pos)
    backup.begin()
    return layer.forward_chunk(inputs,pos,T)

@torch.no_grad()
def validate_layer(layer,index,T,mode):
    samples=[];inputs=torch.empty((T,1,E.HCH),device='npu',dtype=torch.bfloat16)
    pos=torch.tensor([4096],device='npu',dtype=torch.int64);backup=PrefixState(layer,pos,T)
    starts=(4093,4096,4097,8192-T)
    for sample,start in enumerate(starts):
        guard();pos.fill_(start);torch.manual_seed(16100+index*10+T+sample)
        inputs.copy_(torch.randn_like(inputs)*(.003,.03,.3,1.)[sample])
        seed_state(layer,16200+sample);pre=capture_cpu(layer)
        serial=[]
        for t in range(T):serial.append(layer.forward(inputs[t:t+1],pos+t).cpu().clone())
        serial_out=torch.cat(serial);serial_state=capture_cpu(layer)
        load_cpu(layer,pre)
        batched=verify_call(layer,inputs,pos,T,backup).cpu().clone();post=capture_cpu(layer)
        diff=err(serial_out,batched);assert diff['finite']
        state_diff={'conv':err(serial_state['conv'],post['conv'])}
        if 'S' in post:state_diff['S']=err(serial_state['S'],post['S'])
        if layer.ltype in ('linear','ple'):
            sj=small_delta.journals[(layer.S.data_ptr(),T)]
            cj=conv_journals[(layer.S.data_ptr(),T)]
            assert torch.equal(sj[0].cpu(),pre['S']) and torch.equal(sj[-1].cpu(),post['S'])
            assert torch.equal(cj[0].cpu(),pre['conv']) and torch.equal(cj[-1].cpu(),post['conv'])
        next_h=torch.randn((1,1,E.HCH),device='npu',dtype=torch.bfloat16)*.03
        cuts=[]
        for cut in range(T+1):
            guard();expected=expected_prefix(layer,pre,post,backup,cut)
            load_cpu(layer,post);backup.restore(cut);got=capture_cpu(layer)
            assert equal_cpu(got,expected),('PREFIX_STATE_MISMATCH',index,mode,T,start,cut)
            continued=None
            if start+cut<8192:
                actual_next=layer.forward(next_h,pos+cut).cpu().clone();actual_state=capture_cpu(layer)
                load_cpu(layer,expected)
                expected_next=layer.forward(next_h,pos+cut).cpu().clone();expected_state=capture_cpu(layer)
                assert torch.equal(actual_next,expected_next) and equal_cpu(actual_state,expected_state),('CONTINUATION_MISMATCH',index,mode,T,start,cut)
                continued=True
            cuts.append(dict(cut=cut,state_exact=True,continuation_exact=continued))
        samples.append(dict(start=start,amplitude=(.003,.03,.3,1.)[sample],
            batch_vs_serial_output=diff,batch_vs_serial_state=state_diff,prefix_cuts=cuts))
    byte_count=backup.bytes()
    # Graph replay must update position, inputs, output, state and all prefixes.
    stream=torch.npu.Stream();fresh=[]
    with torch.npu.stream(stream):
        seed_state(layer,16300);pos.fill_(4096);verify_call(layer,inputs,pos,T,backup);stream.synchronize()
        graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph,stream=stream):captured=verify_call(layer,inputs,pos,T,backup)
        stream.synchronize()
        for start in (4093,4097):
            guard();pos.fill_(start);inputs.copy_(torch.randn_like(inputs)*.03);seed_state(layer,16300+start)
            pre=capture_cpu(layer)
            eager=verify_call(layer,inputs,pos,T,backup).cpu().clone();expected=capture_cpu(layer)
            if layer.ltype in ('linear','ple'):
                sj=small_delta.journals[(layer.S.data_ptr(),T)].cpu().clone()
                cj=conv_journals[(layer.S.data_ptr(),T)].cpu().clone()
            load_cpu(layer,pre);graph.replay();stream.synchronize()
            assert torch.equal(captured.cpu(),eager) and equal_cpu(capture_cpu(layer),expected),('FRESH_GRAPH_STATE',index,mode,T,start)
            if layer.ltype in ('linear','ple'):
                assert torch.equal(small_delta.journals[(layer.S.data_ptr(),T)].cpu(),sj)
                assert torch.equal(conv_journals[(layer.S.data_ptr(),T)].cpu(),cj)
            fresh.append(start)
        graph.reset()
    return dict(T=T,mode=mode,samples=samples,fresh_graph_positions=fresh,prefix_storage_bytes=byte_count)

@torch.no_grad()
def time_pair(layers,T,index):
    guard();torch.manual_seed(16400+index+T)
    inputs=torch.randn((T,1,E.HCH),device='npu',dtype=torch.bfloat16)*.03
    pos=torch.tensor([4096],device='npu',dtype=torch.int64)
    backups={name:PrefixState(layer,pos,T) if T>1 else None for name,layer in layers.items()}
    stream=torch.npu.Stream();graphs={};times={name:[] for name in layers};outs={}
    torch.npu.empty_cache()
    with torch.npu.stream(stream):
        for name,layer in layers.items():
            seed_state(layer,16500);verify_call(layer,inputs,pos,T,backups[name]);stream.synchronize()
            g=torch.npu.NPUGraph()
            with torch.npu.graph(g,stream=stream,pool=next(iter(graphs.values())).pool() if graphs else None):
                for _ in range(8):out=verify_call(layer,inputs,pos,T,backups[name])
            graphs[name]=g;outs[name]=out;g.replay();stream.synchronize()
        for rep in range(12):
            guard()
            for name in (tuple(layers) if rep%2==0 else tuple(reversed(layers))):
                begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True);begin.record(stream)
                for _ in range(2):graphs[name].replay()
                end.record(stream);end.synchronize();times[name].append(begin.elapsed_time(end)/16)
        for g in graphs.values():g.reset()
    return dict(T=T,median_ms={k:statistics.median(v) for k,v in times.items()},samples_ms=times,
        journal_in_forward=T>1,storage_bytes={name:b.bytes() if b else 0 for name,b in backups.items()})

@torch.no_grad()
def main():
    guard();assert os.environ.get('FNX_INDEX_CACHE')=='1'
    recurrence=small_delta.validate();records=[]
    for index,kind in ((device*24,'linear'),(device*24+3,'full')):
        guard();E._GAMMA_CACHE.clear()
        weights=load_layer(f'{QDIR}/L{index:02d}.safetensors',index,'npu:0');layers={}
        for name in ('bf16','w8a8'):
            os.environ['FNX_DENSE_W8A8']='1' if name=='w8a8' else '0'
            layer=E.FnLayer(index,kind,weights,'npu:0')
            if kind=='full':
                layer.kv=E.KVStore(8192,'npu:0',1)
                layer.index_k=torch.zeros(8192,128,device='npu',dtype=torch.bfloat16)
                layer.index_keys=torch.zeros(2048,128,device='npu',dtype=torch.float32)
            layers[name]=layer
        del weights;torch.npu.empty_cache()
        expected=['w_qkvz','w_out'] if kind=='linear' else ['w_q','w_o']
        assert layers['w8a8'].w8a8_replacements==expected
        projections={}
        for name in expected:
            obj=getattr(layers['w8a8'],name);assert isinstance(obj,SmoothW8A8) and obj.weight.dtype==torch.int8
            key=f'L{index:02d}/{name}';object_names[id(obj)]=key
            projections[name]=dict(type=type(obj).__name__,weight_dtype=str(obj.weight.dtype),shape=list(obj.weight.shape),layout=obj.layout)
        print('PAIRED_LAYER_LOADED',index,torch.npu.memory_allocated(),flush=True)
        validation={}
        for T in (2,4):
            for name,layer in layers.items():
                validation[f'{name}/T{T}']=validate_layer(layer,index,T,name)
                print('PREFIX_AND_GRAPH_VALIDATED',index,name,T,flush=True)
        times=[time_pair(layers,T,index) for T in (1,2,4)]
        row=dict(layer=index,kind=kind,projections=projections,validation=validation,timing=times)
        records.append(row)
        (P/f'layers-device{device}.json').write_text(json.dumps(dict(status='running',records=records),indent=2)+'\n')
        print('V4_LAYER_COST',index,[r['median_ms'] for r in times],flush=True)
        del layers,layer,row,validation,times
        small_delta.journals.clear();conv_journals.clear();object_names.clear()
        E._GAMMA_CACHE.clear();gc.collect();torch.npu.empty_cache()
    result=dict(status='passed_component_proxy',device=device,records=records,independent_recurrence=recurrence,
        smooth_projection_call_rows=call_hist,exclusive=False,production_traffic_stable=True,
        production_transfers=transfers,allocator_fraction=1/24,peak_allocated_bytes=torch.npu.max_memory_allocated(),
        elapsed_s=time.monotonic()-started,context_capacity=8192,synthetic_hidden=True,
        scope='4-layer weighted proxy only; includes prefix journals, excludes drafter/head/IPC/full quality')
    (P/f'layers-device{device}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('R161_PAIRED_VERIFIER_COMPLETE',device,result['elapsed_s'],result['peak_allocated_bytes'],flush=True)

if __name__=='__main__':main()
