import ctypes,gc,json,statistics,time
from pathlib import Path
import torch,torch_npu

lib=ctypes.CDLL('/root/ascend-dev/r104/libfnx_combine.so')
lib.fnx_combine.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*2+[ctypes.c_void_p]
lib.fnx_combine.restype=ctypes.c_int

@torch.no_grad()
def main():
    torch.npu.set_device(0);torch.manual_seed(104104);records=[]
    for T in (1,7,128,512,4096,8192):
        N=T*10;H=2560
        yd=torch.randn(N,H,dtype=torch.bfloat16,device='npu:0')*.1
        inverse=torch.randperm(N,device='npu:0').int()
        weights=torch.softmax(torch.randn(T,10,device='npu:0'),-1).to(torch.bfloat16)
        out=torch.empty(T,H,dtype=yd.dtype,device='npu:0')
        def base():return (yd.index_select(0,inverse.long()).view(T,10,H)*weights.unsqueeze(-1)).sum(1)
        def fused(blocks=40):
            wf=weights.float()
            assert lib.fnx_combine(yd.data_ptr(),inverse.data_ptr(),wf.data_ptr(),out.data_ptr(),T,blocks,torch.npu.current_stream().npu_stream)==0
            return out
        for blocks in (20,40):
            expected=base();fused(blocks);torch.npu.synchronize()
            item={'tokens':T,'blocks':blocks,'exact':torch.equal(out,expected),'max_abs':(out.float()-expected.float()).abs().max().item()}
            print('CUSTOM_COMBINE_CHECK',json.dumps(item),flush=True)
            if not item['exact']:records.append(item);continue
            stream=torch.npu.Stream();graph=torch.npu.NPUGraph()
            with torch.npu.stream(stream):
                with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):fused(blocks)
            torch.npu.synchronize();times=[]
            for _ in range(14):
                start=time.perf_counter();graph.replay();torch.npu.synchronize();times.append((time.perf_counter()-start)*1000)
            item['median_ms']=statistics.median(times[2:]);item['dynamic_exact']=True
            for seed in range(8):
                torch.manual_seed(seed);yd.copy_(torch.randn_like(yd)*(.001 if seed%2 else 1))
                inverse.copy_(torch.randperm(N,device='npu:0').int())
                weights.copy_(torch.softmax(torch.randn(T,10,device='npu:0')*(seed+1),-1).to(torch.bfloat16))
                expected=base();graph.replay();torch.npu.synchronize()
                if not torch.equal(out,expected):
                    item['dynamic_exact']=False;item['max_dynamic_abs']=(out.float()-expected.float()).abs().max().item();break
            records.append(item);print('CUSTOM_COMBINE',json.dumps(item),flush=True)
            del graph;gc.collect();torch.npu.empty_cache()
        graph=torch.npu.NPUGraph();stream=torch.npu.Stream()
        with torch.npu.stream(stream):
            with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):ref=base()
        torch.npu.synchronize();times=[]
        for _ in range(14):
            start=time.perf_counter();graph.replay();torch.npu.synchronize();times.append((time.perf_counter()-start)*1000)
        records.append({'tokens':T,'mode':'inverse','median_ms':statistics.median(times[2:])})
        print('CUSTOM_BASELINE',json.dumps(records[-1]),flush=True)
        del graph,yd,inverse,weights,out,expected,ref;gc.collect();torch.npu.empty_cache()
    Path('/root/ascend-dev/r104/probe.json').write_text(json.dumps(records,indent=2));print('R104_PROBE_COMPLETE',flush=True)

if __name__=='__main__':main()
