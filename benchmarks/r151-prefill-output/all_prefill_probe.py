"""Online, bounded real-weight DeltaNet checks; eager and graph replay.

Loads only DeltaNet projections, not full layers, MoE, KV or an Engine.
"""
import gc,hashlib,json,os,sys,time,urllib.request
from pathlib import Path
import torch,torch_npu
from safetensors import safe_open
sys.path.insert(0,str(Path(__file__).resolve().parent/'candidate'))
from fnx import engine as E,prefill_output
from fnx.load import QDIR,SRCDIR
P=Path(__file__).resolve().parent;physical=int(sys.argv[1]);started=time.monotonic()
torch.set_num_threads(1);torch.npu.set_device(0)
torch.npu.set_per_process_memory_fraction(1/64,0);torch.manual_seed(14720+physical)
assert Path(E.__file__).resolve()==Path('/models/fnx-diagnostics/r151-prefill-post-20261112/candidate/fnx/engine.py')
fused_calls=0
original_fused=prefill_output.add_into
def counted_fused(*args,**kwargs):
    global fused_calls
    result=original_fused(*args,**kwargs)
    assert result,('FUSED_FALLBACK',[(x.shape,x.dtype,x.stride()) for x in args if isinstance(x,torch.Tensor)])
    fused_calls+=1
    return result
prefill_output.add_into=counted_fused
def guard():
    h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
    assert h['service_revision']=='r150' and h['worker_alive']
    assert torch.npu.mem_get_info()[0]>2048*1024**2,'online HBM headroom guard'
    assert time.monotonic()-started<600,'bounded probe elapsed guard'
def compare_cpu(a,b):
    aa=a.detach().contiguous().view(torch.uint8);bb=b.detach().contiguous().view(torch.uint8)
    return dict(exact=torch.equal(aa,bb),different_bytes=int((aa!=bb).sum()),bytes=aa.numel())
def tensor_compare(a,b):return compare_cpu(a.detach().cpu(),b.detach().cpu())
def load_weight(file,layer,name):
    key=f'model.language_model.layers.{layer}.linear_attn.{name}'
    if key in file.keys():return file.get_tensor(key).npu()
    q=file.get_tensor(key+'.q8').npu();s=file.get_tensor(key+'.s8').npu()
    pad=int(file.get_tensor(key+'.__pad').item()) if key+'.__pad' in file.keys() else 0
    if q.ndim==2 and os.environ.get('FNX_DENSE_W8')=='1':
        return E.DenseW8(q,s,K=q.shape[-1]-pad).deq_w()
    return E.QW(q,s,K=q.shape[-1]-pad).deq_w().contiguous()
def save(records,status,**extra):
    result=dict(status=status,physical_device=physical,layers=records,scope='FnLayer._deltanet_chunk prefill, T65/128/512 and dynamic graph',
        hidden_inputs='synthetic; four amplitudes',uses_real_weights=True,weight_loader='DenseW8.deq_w / raw tensors, matching production',
        engine_module=E.__file__,allocator_limit_fraction=1/64,max_allocated_bytes=torch.npu.max_memory_allocated(),
        max_reserved_bytes=torch.npu.max_memory_reserved(),elapsed_s=time.monotonic()-started,
        fused_python_calls=fused_calls,service_kept_online=True,exclusive=False,**extra)
    (P/f'all-linear-device{physical}.json').write_text(json.dumps(result,indent=2))
@torch.no_grad()
def main():
    guard();records=[]
    cfg=json.load(open(Path(SRCDIR)/'config.json'))['text_config']['layer_types']
    indices=[i for i,t in enumerate(cfg) if t not in E.FULL_TYPES and (i<24)==(physical==0)]
    assert len(indices)==18,indices
    for layer in indices:
        guard();E._GAMMA_CACHE.clear();attrs={};before_calls=fused_calls
        with safe_open(f'{QDIR}/L{layer:02d}.safetensors',framework='pt',device='cpu') as file:
            for dest,name in [('w_qkv','in_proj_qkv.weight'),('w_z','in_proj_z.weight'),('w_a','in_proj_a.weight'),('w_b','in_proj_b.weight'),('w_out','out_proj.weight')]:attrs[dest]=E.Raw(load_weight(file,layer,name))
            attrs['conv_w']=load_weight(file,layer,'conv1d.weight').view(E.CONV_DIM,4)
            attrs['neg_exp_a']=-torch.exp(load_weight(file,layer,'A_log').float())
            attrs['dt_bias']=load_weight(file,layer,'dt_bias').float()
            attrs['dn_norm_w']=load_weight(file,layer,'norm.weight').float()
            attrs['dn_norm_g']=(attrs['dn_norm_w']-1.).contiguous()
        history=torch.randn(1,E.CONV_DIM,3,dtype=torch.bfloat16,device='npu')*.01
        state=torch.randn(1,E.NVH,E.HKD,E.HVD,dtype=torch.float32,device='npu')*.01
        branches={}
        for mode in ('baseline','candidate'):
            obj=E.FnLayer.__new__(E.FnLayer);obj.__dict__.update(attrs,w_qkvz=None,w_ab=None,qkvz_ba=False,_dn_lens={},dev='npu:0',idx=layer,ltype='linear')
            obj.conv=history.clone();obj.S=state.clone();branches[mode]=obj
        assert branches['baseline'].S.data_ptr()!=branches['candidate'].S.data_ptr()
        tests=[]
        def check(step,kind,outputs):
            a,b=branches['baseline'],branches['candidate']
            comparison=dict(output=compare_cpu(outputs['baseline'],outputs['candidate']),conv_history=tensor_compare(a.conv,b.conv),fp32_state=tensor_compare(a.S,b.S))
            row=dict(mode=kind,step=step,comparison=comparison);tests.append(row)
            if not all(x['exact'] for x in comparison.values()):
                records.append(dict(layer=layer,tests=tests,all_bitwise_equal=False));save(records,'rejected');raise AssertionError(('PREFILL_REJECT',physical,layer,row))
        for T in (65,128,512):
            guard();hidden=torch.randn(T,1,E.H,dtype=torch.bfloat16,device='npu')*.03;outputs={}
            for mode,obj in branches.items():
                os.environ['FNX_PREFILL_OUTPUT']='1' if mode=='candidate' else '0'
                outputs[mode]=obj._deltanet_chunk(hidden,T).detach().cpu();time.sleep(.03)
            check(T,'eager',outputs)
        snapshots={mode:(obj.conv.clone(),obj.S.clone()) for mode,obj in branches.items()}
        hidden=torch.randn(128,1,E.H,dtype=torch.bfloat16,device='npu')*.03;graph_input=torch.empty_like(hidden);graph_input.copy_(hidden)
        stream=torch.npu.Stream();torch.npu.synchronize();graphs={};graph_outputs={}
        for mode,obj in branches.items():
            os.environ['FNX_PREFILL_OUTPUT']='1' if mode=='candidate' else '0';g=torch.npu.NPUGraph()
            with torch.npu.stream(stream):
                with torch.npu.graph(g,stream=stream):graph_outputs[mode]=obj._deltanet_chunk(graph_input,128)
            stream.synchronize();graphs[mode]=g
            obj.conv.copy_(snapshots[mode][0]);obj.S.copy_(snapshots[mode][1])
        torch.npu.synchronize()
        for step in range(4):
            guard();graph_input.copy_(torch.randn_like(hidden)*[.003,.03,.3,1.][step]);torch.npu.synchronize();outputs={}
            for mode in ('baseline','candidate'):
                with torch.npu.stream(stream):graphs[mode].replay()
                stream.synchronize();outputs[mode]=graph_outputs[mode].detach().cpu()
            check(step,'dynamic_graph',outputs);time.sleep(.03)
        assert fused_calls-before_calls==14,(layer,fused_calls-before_calls)
        records.append(dict(layer=layer,tests=tests,all_bitwise_equal=True,native_calls=fused_calls-before_calls));save(records,'running')
        print('REAL_PREFILL_LAYER_PASS',physical,layer,len(tests),fused_calls-before_calls,flush=True)
        for g in graphs.values():g.reset()
        del graphs,graph_outputs,snapshots,branches,attrs,obj,outputs,hidden,graph_input
        E._GAMMA_CACHE.clear();gc.collect();torch.npu.empty_cache()
    save(records,'passed');print('ALL_REAL_PREFILL_COMPLETE',physical,len(records),flush=True)
if __name__=='__main__':main()
