"""Real layers, dynamic decode and actual prefill, output/history/FP32 state bits."""
import gc,json,os,statistics
from pathlib import Path
import torch,torch_npu
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'candidate'))
from fnx import engine as E
from fnx.load import load_layer,QDIR
P=Path(__file__).resolve().parent
def equal(a,b):
    aa=a.cpu().contiguous().view(torch.uint8);bb=b.cpu().contiguous().view(torch.uint8)
    return dict(exact=torch.equal(aa,bb),different_bytes=int((aa!=bb).sum()),bytes=aa.numel())
@torch.no_grad()
def main():
    torch.manual_seed(1464);records=[]
    for index,device in [(0,'npu:0'),(12,'npu:0'),(24,'npu:1'),(44,'npu:1')]:
        torch.npu.set_device(device)
        E._GAMMA_CACHE.clear()
        capture_stream=torch.npu.Stream()
        W=load_layer(f'{QDIR}/L{index:02d}.safetensors',index,device)
        layers={mode:E.FnLayer(index,'linear',W,device) for mode in ('baseline','candidate')}
        def state(L):return [L.conv,L.S]+([L.ple_conv] if getattr(L,'ple_conv',None) is not None else [])
        initial=[]
        for t in state(layers['baseline']):t.normal_(std=.01);initial.append(t.clone())
        pos=torch.tensor([8192],dtype=torch.int64,device=device)
        for T in (1,128,512,2048):
            inputs=[torch.randn(T,1,E.HCH,dtype=torch.bfloat16,device=device)*.03 for _ in range(24 if T==1 else 4)]
            results={};graphs={}
            for mode,L in layers.items():
                os.environ['FNX_PREFILL_OUTPUT']='1' if mode=='candidate' else '0'
                for a,b in zip(state(L),initial):a.copy_(b)
                observations=[]
                for x in inputs:
                    out=L.forward(x,pos) if T==1 else L.forward_chunk(x,pos,T)
                    observations.append([out.clone()]+[t.clone() for t in state(L)])
                torch.npu.synchronize();results[mode]=observations
                if T==1:
                    g=torch.npu.NPUGraph()
                    with torch.npu.stream(capture_stream):
                        with torch.npu.graph(g,stream=capture_stream):
                            for _ in range(8):output=L.forward(inputs[0],pos)
                    graphs[mode]=g
            capture_stream.synchronize()
            comparisons=[[equal(a,b) for a,b in zip(aa,bb)] for aa,bb in zip(results['baseline'],results['candidate'])]
            assert all(x['exact'] for row in comparisons for x in row),(index,T,comparisons)
            times={name:[] for name in graphs}
            for round in range(10):
                for name in (list(graphs) if round%2==0 else list(reversed(graphs))):
                    with torch.npu.stream(capture_stream):
                        g=graphs[name];a=torch.npu.Event(enable_timing=True);b=torch.npu.Event(enable_timing=True);a.record(capture_stream)
                        for _ in range(4):g.replay()
                        b.record(capture_stream);b.synchronize();times[name].append(a.elapsed_time(b)/32)
            row=dict(layer=index,device=device,T=T,steps=len(inputs),comparison_order=['output','conv','fp32_state']+(['ple_conv'] if len(initial)>2 else []),comparisons=comparisons,times_ms=times,median_ms={k:statistics.median(v) for k,v in times.items()})
            records.append(row);(P/'layer-probe.json').write_text(json.dumps(records,indent=2));print('REAL_LAYER_PASS',index,T,row['median_ms'],flush=True)
            for g in graphs.values():g.reset()
            del graphs,results,observations,inputs
            gc.collect();torch.npu.empty_cache()
        del layers,W,L,initial;E._GAMMA_CACHE.clear();gc.collect();torch.npu.empty_cache()
    print('REAL_LAYER_COMPLETE',flush=True)
if __name__=='__main__':main()
