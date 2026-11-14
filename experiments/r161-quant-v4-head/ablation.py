"""Same-run journal ablation; full prefix tests are repeated for packed mode."""
import gc,json,os,statistics,time
from pathlib import Path
import torch,torch_npu
import probe as B
from fnx import engine as E
from fnx.load import load_layer,QDIR
from fnx.smooth_w8a8 import SmoothW8A8
import small_delta
from verifier_state import PrefixState,conv_journals
import packed_journal
P=Path(__file__).resolve().parent;device=B.device
packed_journal.install()

@torch.no_grad()
def timing(layers,index):
    torch.manual_seed(16600+index);T=4
    inputs=torch.randn((T,1,E.HCH),device='npu',dtype=torch.bfloat16)*.03
    pos=torch.tensor([4096],device='npu',dtype=torch.int64)
    variants=[(model,journal) for model in layers for journal in ('legacy','loop','packed')]
    backups={(m,j):PrefixState(layers[m],pos,T) for m,j in variants}
    graphs={};times={m+'/'+j:[] for m,j in variants};stream=torch.npu.Stream()
    torch.npu.empty_cache()
    with torch.npu.stream(stream):
        for model,journal in variants:
            layer=layers[model];backup=backups[(model,journal)]
            os.environ['R161_CONV_MODE']=journal;B.seed_state(layer,16700)
            def fn():
                if journal!='legacy':backup.begin()
                return layer.forward_chunk(inputs,pos,T)
            fn();stream.synchronize();g=torch.npu.NPUGraph()
            with torch.npu.graph(g,stream=stream,pool=next(iter(graphs.values())).pool() if graphs else None):
                for _ in range(8):out=fn()
            graphs[model+'/'+journal]=g;g.replay();stream.synchronize()
        for rep in range(12):
            B.guard()
            for name in (tuple(graphs) if rep%2==0 else tuple(reversed(graphs))):
                begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True);begin.record(stream)
                for _ in range(2):graphs[name].replay()
                end.record(stream);end.synchronize();times[name].append(begin.elapsed_time(end)/16)
        for g in graphs.values():g.reset()
    return dict(T=4,median_ms={k:statistics.median(v) for k,v in times.items()},samples_ms=times,
        legacy='FP32 S prefix only; incomplete rollback',loop='S+loop conv prefix or KV/index tails',
        packed='S+single-copy conv prefix or KV/index tails')

@torch.no_grad()
def main():
    B.guard();records=[]
    for index,kind in ((device*24,'linear'),(device*24+3,'full')):
        E._GAMMA_CACHE.clear();weights=load_layer(f'{QDIR}/L{index:02d}.safetensors',index,'npu:0');layers={}
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
        for name in expected:
            obj=getattr(layers['w8a8'],name);assert isinstance(obj,SmoothW8A8) and obj.weight.dtype==torch.int8
            B.object_names[id(obj)]=f'L{index:02d}/{name}'
        packed_equivalence={}
        if kind=='linear':packed_equivalence={m:packed_journal.validate(layer) for m,layer in layers.items()}
        os.environ['R161_CONV_MODE']='packed';validation={}
        for T in (2,4):
            for name,layer in layers.items():
                validation[f'{name}/T{T}']=B.validate_layer(layer,index,T,name)
                print('PACKED_PREFIX_VALIDATED',index,name,T,flush=True)
        row=dict(layer=index,kind=kind,packed_conv_equivalence=packed_equivalence,
            validation=validation,timing=timing(layers,index))
        records.append(row)
        (P/f'ablation-device{device}.json').write_text(json.dumps(dict(status='running',records=records),indent=2)+'\n')
        print('JOURNAL_ABLATION',index,row['timing']['median_ms'],flush=True)
        del layers,layer,validation,row
        small_delta.journals.clear();conv_journals.clear();B.object_names.clear();E._GAMMA_CACHE.clear()
        gc.collect();torch.npu.empty_cache()
    result=dict(status='passed_packed_journal_component',device=device,records=records,
        exclusive=False,production_traffic_stable=True,smooth_projection_call_rows=B.call_hist,
        elapsed_s=time.monotonic()-B.started,peak_allocated_bytes=torch.npu.max_memory_allocated(),
        allocator_fraction=1/24,scope='same-run four-layer proxy; no drafter/head/IPC/full quality')
    (P/f'ablation-device{device}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('R161_ABLATION_COMPLETE',device,result['elapsed_s'],flush=True)

if __name__=='__main__':main()
