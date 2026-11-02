"""Real-weight MoE equality including routing and captured input changes."""
import gc,json,os
from pathlib import Path
import torch,torch_npu
from fnx import engine as E
from fnx.load import load_layer,QDIR

@torch.no_grad()
def main():
    torch.npu.set_device(0);torch.manual_seed(100100)
    W=load_layer(os.path.join(QDIR,'L00.safetensors'),0,'npu:0');L=E.FnLayer(0,'linear',W,'npu:0');del W
    results=[]
    for T in (128,512,8192):
        x=torch.randn(T,1,E.H,dtype=torch.bfloat16,device='npu:0')*.03
        os.environ['FNX_MOE_INVERSE']='0';baseline=L._moe(x).clone()
        os.environ['FNX_MOE_INVERSE']='1';actual=L._moe(x)
        assert torch.equal(actual.view(torch.int16),baseline.view(torch.int16)),('eager',T)
        stream=torch.npu.Stream();graph=torch.npu.NPUGraph()
        with torch.npu.stream(stream):
            with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):actual=L._moe(x)
        torch.npu.synchronize()
        for seed in (1,2,3):
            torch.manual_seed(seed);x.copy_(torch.randn_like(x)*.03)
            graph.replay();torch.npu.synchronize();saved=actual.clone()
            os.environ['FNX_MOE_INVERSE']='0';reference=L._moe(x)
            assert torch.equal(saved.view(torch.int16),reference.view(torch.int16)),('graph',T,seed)
            os.environ['FNX_MOE_INVERSE']='1';del saved,reference
        del graph,actual,baseline,x;gc.collect();torch.npu.empty_cache()
        results.append({'tokens':T,'real_layer':0,'eager_exact':True,'dynamic_graph_seeds':[1,2,3]})
        print('REAL_MOE_EXACT',T,flush=True)
    Path('/root/ascend-dev/r100/moe-test.json').write_text(json.dumps(results,indent=2))
    print('R100_MOE_ACCEPT',flush=True)

if __name__=='__main__':main()
