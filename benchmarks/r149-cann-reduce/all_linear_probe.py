"""Online, bounded real-weight DeltaNet checks; eager and graph replay.

Loads only DeltaNet projections, not full layers, MoE, KV or an Engine.
"""
import gc,hashlib,json,os,sys,time,urllib.request
from pathlib import Path
import torch,torch_npu
from safetensors import safe_open
sys.path.insert(0,str(Path(__file__).resolve().parent/'candidate'))
from fnx import engine as E,norm_pair
from fnx.load import QDIR,SRCDIR
P=Path(__file__).resolve().parent;physical=int(sys.argv[1]);started=time.monotonic()
torch.set_num_threads(1);torch.npu.set_device(0)
torch.npu.set_per_process_memory_fraction(1/128,0);torch.manual_seed(14720+physical)
assert Path(E.__file__).resolve()==Path('/models/fnx-diagnostics/r149-cann-reduce-20261112/candidate/fnx/engine.py')
fused_calls=0
original_fused=norm_pair.run
def counted_fused(values,**kwargs):
    global fused_calls
    result=original_fused(values,**kwargs)
    assert result is not None,('FUSED_FALLBACK',values.shape,values.dtype,values.stride())
    fused_calls+=1
    return result
norm_pair.run=counted_fused
def guard():
    h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
    assert h['service_revision']=='r144' and h['worker_alive'] and not h.get('norm_pair',False)
    assert torch.npu.mem_get_info()[0]>1536*1024**2,'online HBM headroom guard'
    assert time.monotonic()-started<340,'bounded probe elapsed guard'
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
    result=dict(status=status,physical_device=physical,layers=records,scope='FnLayer._deltanet only',
        hidden_inputs='synthetic; four amplitudes',uses_real_weights=True,weight_loader='DenseW8.deq_w / raw tensors, matching production',
        engine_module=E.__file__,allocator_limit_fraction=1/128,max_allocated_bytes=torch.npu.max_memory_allocated(),
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
            obj=E.FnLayer.__new__(E.FnLayer);obj.__dict__.update(attrs,w_qkvz=None,w_ab=None,qkvz_ba=False,dev='npu:0',idx=layer,ltype='linear')
            obj.conv=history.clone();obj.S=state.clone();branches[mode]=obj
        assert branches['baseline'].S.data_ptr()!=branches['candidate'].S.data_ptr()
        tests=[]
        def check(step,kind,outputs):
            a,b=branches['baseline'],branches['candidate']
            comparison=dict(output=compare_cpu(outputs['baseline'],outputs['candidate']),conv_history=tensor_compare(a.conv,b.conv),fp32_state=tensor_compare(a.S,b.S))
            row=dict(mode=kind,step=step,comparison=comparison);tests.append(row)
            if not all(x['exact'] for x in comparison.values()):
                torch.save(dict(input=hidden.cpu(),baseline_output=outputs['baseline'],candidate_output=outputs['candidate'],baseline_state=a.S.cpu(),candidate_state=b.S.cpu()),P/f'rejected-fullscan-{physical}-{layer}-{kind}-{step}.pt')
                records.append(dict(layer=layer,tests=tests,all_bitwise_equal=False));save(records,'rejected');raise AssertionError(('DELTANET_BRANCH_REJECT',physical,layer,kind,step,comparison))
        for step in range(16):
            guard();hidden=torch.randn(1,1,E.H,dtype=torch.bfloat16,device='npu')*[.003,.03,.3,1.][step%4];outputs={}
            for mode,obj in branches.items():
                os.environ['FNX_NORM_PAIR']='1' if mode=='candidate' else '0'
                outputs[mode]=obj._deltanet(hidden).detach().cpu();time.sleep(.015)
            check(step,'eager',outputs);time.sleep(.03)
        # Both branches now have identical recurrent state. Capture once, then
        # restore it so capture-time execution does not contaminate replay.
        snapshots={mode:(obj.conv.clone(),obj.S.clone()) for mode,obj in branches.items()}
        graph_input=torch.empty_like(hidden);graph_input.copy_(hidden)
        torch.npu.synchronize();stream=torch.npu.Stream();graphs={};graph_outputs={}
        for mode,obj in branches.items():
            os.environ['FNX_NORM_PAIR']='1' if mode=='candidate' else '0'
            graph=torch.npu.NPUGraph()
            with torch.npu.stream(stream):
                with torch.npu.graph(graph,stream=stream):graph_outputs[mode]=obj._deltanet(graph_input)
            graphs[mode]=graph;stream.synchronize();time.sleep(.02)
        for mode,obj in branches.items():obj.conv.copy_(snapshots[mode][0]);obj.S.copy_(snapshots[mode][1])
        torch.npu.synchronize()
        for step in range(8):
            guard();hidden=torch.randn_like(graph_input)*[.003,.03,.3,1.][step%4];graph_input.copy_(hidden);torch.npu.synchronize();outputs={}
            for mode,graph in graphs.items():
                with torch.npu.stream(stream):graph.replay()
                stream.synchronize();outputs[mode]=graph_outputs[mode].detach().cpu();time.sleep(.015)
            check(step,'graph_replay',outputs);time.sleep(.03)
        assert fused_calls-before_calls==17,('FUSED_BRANCH_CALL_COUNT',layer,fused_calls-before_calls)
        row=dict(layer=layer,tests=tests,all_bitwise_equal=True,eager_steps=16,graph_replay_steps=8,fused_python_calls=17)
        records.append(row);save(records,'in_progress');print('ALL_LINEAR_LAYER_PASS',physical,layer,'eager16+graph8',flush=True)
        for graph in graphs.values():graph.reset()
        del graphs,graph,graph_outputs,graph_input,stream,snapshots,branches,attrs,obj,outputs,history,state,hidden
        E._GAMMA_CACHE.clear();gc.collect();torch.npu.empty_cache();time.sleep(.1)
    guard();save(records,'passed',expected_layers=indices);print('ALL_LINEAR_BRANCHES_COMPLETE',physical,len(records),flush=True)
if __name__=='__main__':main()
