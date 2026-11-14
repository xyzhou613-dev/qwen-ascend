"""Hot single-layer prefix restore cost; no CPU decision/RPC/full-network claim."""
import gc,json,os,statistics,time
from pathlib import Path
import torch,torch_npu
import probe as B
from fnx import engine as E
from fnx.load import load_layer,QDIR
from fnx.smooth_w8a8 import SmoothW8A8
import small_delta
from verifier_state import PrefixState,capture_cpu,load_cpu,equal_cpu,expected_prefix,conv_journals
P=Path(__file__).resolve().parent;device=B.device

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
        del weights;torch.npu.empty_cache();modes={}
        for name,layer in layers.items():
            expected_names=['w_qkvz','w_out'] if kind=='linear' else ['w_q','w_o']
            if name=='w8a8':
                for attr in expected_names:
                    obj=getattr(layer,attr);assert isinstance(obj,SmoothW8A8)
                    B.object_names[id(obj)]=f'L{index:02d}/{attr}'
            B.guard();torch.manual_seed(16900+index)
            h=torch.randn((4,1,E.HCH),device='npu',dtype=torch.bfloat16)*.03
            pos=torch.tensor([4097],device='npu',dtype=torch.int64)
            B.seed_state(layer,17000);before=capture_cpu(layer);backup=PrefixState(layer,pos,4)
            B.verify_call(layer,h,pos,4,backup);after=capture_cpu(layer)
            times={};stream=torch.npu.Stream()
            for cut in range(4):
                B.guard();load_cpu(layer,after)
                expected=expected_prefix(layer,before,after,backup,cut)
                with torch.npu.stream(stream):
                    backup.restore(cut);stream.synchronize()
                    assert equal_cpu(capture_cpu(layer),expected)
                    graph=torch.npu.NPUGraph()
                    with torch.npu.graph(graph,stream=stream):
                        for _ in range(16):backup.restore(cut)
                    graph.replay();stream.synchronize()
                    assert equal_cpu(capture_cpu(layer),expected)
                    samples=[]
                    for rep in range(12):
                        B.guard();begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
                        begin.record(stream)
                        for _ in range(2):graph.replay()
                        end.record(stream);end.synchronize();samples.append(begin.elapsed_time(end)/32)
                    graph.reset()
                times[str(cut)]=dict(median_ms=statistics.median(samples),samples_ms=samples,graph_state_exact=True)
            times['4']=dict(median_ms=0,device_ops=0,note='accept full batch: Python restore returns without device work')
            modes[name]=dict(cuts=times,prefix_storage_bytes=backup.bytes())
        row=dict(layer=index,kind=kind,modes=modes);records.append(row)
        print('ROLLBACK_COMPONENT',index,{m:{k:v['median_ms'] for k,v in r['cuts'].items()} for m,r in modes.items()},flush=True)
        (P/f'rollback-device{device}.json').write_text(json.dumps(dict(status='running',records=records),indent=2)+'\n')
        del layers,layer,h,pos,backup
        small_delta.journals.clear();conv_journals.clear();B.object_names.clear();E._GAMMA_CACHE.clear();gc.collect();torch.npu.empty_cache()
    result=dict(status='passed_cached_restore_component',device=device,records=records,exclusive=False,
        production_traffic_stable=True,elapsed_s=time.monotonic()-B.started,peak_allocated_bytes=torch.npu.max_memory_allocated(),
        scope='same state buffers repeated; cache-favorable restore proxy; excludes CPU cut decision/RPC/full network')
    (P/f'rollback-device{device}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('ROLLBACK_PROBE_COMPLETE',device,result['elapsed_s'],flush=True)

if __name__=='__main__':main()
