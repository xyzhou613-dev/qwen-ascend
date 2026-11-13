"""Exclusive real-weight down+epilogue comparison, with all expert IDs covered."""
import ctypes, json, os, statistics, sys, time
from pathlib import Path
import torch, torch_npu
from safetensors import safe_open
from fnx import engine as E
from fnx import kmajor_down as D
from fnx.moe_epilogue import merge as production_merge

P=Path(__file__).resolve().parent
device=int(sys.argv[1]); started=time.monotonic()
torch.set_num_threads(1); torch.manual_seed(15900+device)
torch.npu.set_device(0); torch.npu.set_per_process_memory_fraction(1/32,0)

def guard():
    assert (P/'lease.active').exists() and time.monotonic()-started<400
    assert time.time()<float((P/'window-start.txt').read_text())+2500
    for p in Path('/proc').glob('[0-9]*'):
        try: cmd=p.joinpath('cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError,ProcessLookupError): continue
        assert b'fnx.server' not in cmd and b'fnx.pipeline_worker' not in cmd, 'REFUSED_SHARED_TIMING'

def error(a,b):
    a=a.cpu().float();b=b.cpu().float()
    return dict(relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),
        max_abs=float((a-b).abs().max()),exact_fraction=float((a==b).float().mean()),finite=bool(torch.isfinite(b).all()))

@torch.no_grad()
def main():
    guard(); layer=24*device
    cfg=json.loads(Path('/models/fnx-diagnostics/r155-smooth-tpot-ab-20261113/production-selected-config.json').read_text())
    with safe_open(cfg['FNX_W4A16_DIR']+f'-em/L{layer:02d}.w4.safetensors',framework='pt',device='cpu') as f:
        key='mlp.experts.down_proj'
        obj=E.ExpW4(f.get_tensor(key+'.w4').npu(),f.get_tensor(key+'.s8t').npu(),f.get_tensor(key+'.base').npu(),64,expert_major=True)
    # This probe constructs no FnLayer, so register its single audited weight explicitly.
    from fnx import audit as A
    A.pointer_layers[obj.w4.data_ptr()]=layer
    x=torch.empty((10,640),device='npu',dtype=torch.bfloat16)
    ids=torch.arange(10,device='npu',dtype=torch.int64)
    weights=torch.softmax(torch.randn(1,10,device='npu'),-1).bfloat16()
    shared=torch.empty((1,2560),device='npu',dtype=torch.bfloat16)
    gate=torch.zeros((1,1),device='npu',dtype=torch.bfloat16)
    x.zero_(); shared.zero_(); D.merge(obj,x,ids,weights,shared,gate)
    oldlib=D._lib
    def old_diagonal(ix):
        out=torch.empty((10,2560),device='npu',dtype=torch.bfloat16)
        assert oldlib.fnx_w4_kmajor(obj.w4.data_ptr(),obj._scales.data_ptr(),ix.data_ptr(),x.data_ptr(),out.data_ptr(),256,torch.npu.current_stream().npu_stream)==0
        return out
    def run(label,ix):
        if label=='production':return production_merge(obj.apply(x,ix),weights,shared,gate)
        if label=='old':
            diagonal=old_diagonal(ix); out=torch.empty_like(shared)
            assert oldlib.fnx_diag_epilogue(diagonal.data_ptr(),weights.data_ptr(),shared.data_ptr(),gate.data_ptr(),out.data_ptr(),torch.npu.current_stream().npu_stream)==0
            return out
        os.environ['FNX_BF16_DOWN_TILE']=label.removeprefix('rounded')
        return D.merge(obj,x,ix,weights,shared,gate)
    labels=('production','old','rounded128','rounded256')
    records=[]; covered=set()
    for sample in range(53):
        guard(); ix=(torch.arange(10)+sample*10)%512
        if sample==52:ix.fill_(511)
        covered.update(ix.tolist());ids.copy_(ix.npu())
        amplitude=(.003,.03,.3,1.)[sample%4]
        x.copy_(torch.randn_like(x)*amplitude);shared.copy_(torch.randn_like(shared)*amplitude)
        weights.copy_(torch.softmax(torch.randn(1,10,device='npu'),-1).bfloat16())
        gate.fill_((-15,-2,0,2,15)[sample%5])
        ref=run('production',ids); row=dict(sample=sample,amplitude=amplitude,merge={})
        for label in labels[1:]:
            e=error(ref,run(label,ids)); assert e['finite']
            if label.startswith('rounded'): assert e['relative_l2']<.001, (label,e)
            row['merge'][label]=e
        diagonal_ref=obj.apply_diag(x,ids); row['diagonal']={}
        for tile in (128,256):
            os.environ['FNX_BF16_DOWN_TILE']=str(tile)
            e=error(diagonal_ref,D.apply(obj,x,ids));assert e['finite'] and e['relative_l2']<.001,e
            row['diagonal'][str(tile)]=e
        records.append(row)
    assert len(covered)==512
    # Cover the whole weight bank rather than repeatedly hitting ten cached experts.
    banks=torch.stack([(torch.arange(10)+i*10)%512 for i in range(64)]).npu()
    stream=torch.npu.Stream(); graphs={}; samples={label:[] for label in labels}; fresh={}
    torch.npu.empty_cache()
    with torch.npu.stream(stream):
        for label in labels:
            run(label,banks[0]);stream.synchronize();g=torch.npu.NPUGraph()
            with torch.npu.graph(g,stream=stream,pool=next(iter(graphs.values())).pool() if graphs else None):
                for bank in banks:out=run(label,bank)
            graphs[label]=g;g.replay();stream.synchronize()
        for rep in range(16):
            guard()
            for label in (labels if rep%2==0 else tuple(reversed(labels))):
                begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                begin.record(stream)
                for _ in range(2): graphs[label].replay()
                end.record(stream);end.synchronize()
                samples[label].append(begin.elapsed_time(end)*1000/128)
        for g in graphs.values():g.reset()
        for label in labels[1:]:
            g=torch.npu.NPUGraph()
            with torch.npu.graph(g,stream=stream):captured=run(label,ids)
            stream.synchronize()
            for rep in range(8):
                ids.copy_(torch.randint(0,512,(10,),device='npu'))
                x.copy_(torch.randn_like(x)*.03);shared.copy_(torch.randn_like(shared)*.03)
                weights.copy_(torch.softmax(torch.randn_like(weights.float()),-1).bfloat16())
                gate.fill_(rep-4)
                eager=run(label,ids);g.replay();stream.synchronize()
                assert torch.equal(eager.cpu(),captured.cpu()),label
            fresh[label]=True;g.reset()
    result=dict(status='passed_component',scope='exclusive down+epilogue; not HTTP or full quality',
        device=device,layer=layer,experts_covered=len(covered),median_us={k:statistics.median(v) for k,v in samples.items()},
        samples_us=samples,records=records,fresh_input_graph_verified=fresh,
        peak_allocated_bytes=torch.npu.max_memory_allocated(),elapsed_s=time.monotonic()-started)
    (P/f'component-device{device}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('COMPONENT_RESULT',device,result['median_us'],flush=True)

if __name__=='__main__':main()
