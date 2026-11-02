import os,ctypes,gc,json,statistics,time
from pathlib import Path
import torch,torch_npu

lib=ctypes.CDLL(os.environ.get('FNX_DN_NATIVE_SO',str(Path(__file__).resolve().parents[1]/'fnx/libfnx_delta.so')))
lib.fnx_delta.argtypes=[ctypes.c_void_p]*7+[ctypes.c_int,ctypes.c_void_p]
lib.fnx_delta.restype=ctypes.c_int
def bits_equal(a,b):return torch.equal(a.view(torch.int32),b.view(torch.int32))

@torch.no_grad()
def main():
    torch.npu.set_device(0);torch.manual_seed(108108);records=[]
    initial=torch.randn(1,48,128,128,device='npu:0')*.01
    s=initial.clone();q=torch.randn(1,48,128,dtype=torch.bfloat16,device='npu:0')*.1
    k=torch.randn_like(q)*.08;v=torch.randn_like(q)*.2
    beta=torch.rand(1,48,1,device='npu:0').to(torch.bfloat16)
    decay=torch.full((1,48,1,1),.9,device='npu:0');out=torch.empty(1,48,128,device='npu:0')
    def baseline(state=s):
        memory=state*decay
        change=(v-(memory*k.unsqueeze(-1)).sum(-2))*beta
        updated=memory+k.unsqueeze(-1)*change.unsqueeze(-2)
        result=(updated*q.unsqueeze(-1)).sum(-2)
        state.copy_(updated);return result
    def fused(blocks):
        qf,kf,vf,bf=q.float(),k.float(),v.float(),beta.float()
        assert lib.fnx_delta(s.data_ptr(),qf.data_ptr(),kf.data_ptr(),vf.data_ptr(),bf.data_ptr(),decay.data_ptr(),out.data_ptr(),blocks,torch.npu.current_stream().npu_stream)==0
        return out
    for blocks in (20,24,40):
        q.copy_(torch.randn_like(q)*.1);k.copy_(torch.randn_like(k)*.08);v.copy_(torch.randn_like(v)*.2)
        beta.copy_(torch.rand_like(beta));decay.fill_(.9)
        s.copy_(initial);ref=initial.clone();expected=baseline(ref);fused(blocks);torch.npu.synchronize()
        item={'blocks':blocks,'state_exact':bits_equal(s,ref),'output_exact':bits_equal(out,expected),
              'max_state_abs':(s-ref).abs().max().item(),'max_output_abs':(out-expected).abs().max().item()}
        item['unchanged_heads']=torch.nonzero((s-initial).abs().sum((0,2,3))==0).flatten().cpu().tolist()
        item['wrong_heads']=torch.nonzero((s-ref).abs().sum((0,2,3))!=0).flatten().cpu().tolist()
        print('DELTA_FIRST',json.dumps(item),flush=True)
        if not item['state_exact'] or not item['output_exact']:records.append(item);continue
        stream=torch.npu.Stream();graph=torch.npu.NPUGraph()
        with torch.npu.stream(stream):
            with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):fused(blocks)
        s.copy_(initial);ref.copy_(initial);exact=True
        for i in range(256):
            q.copy_(torch.randn_like(q)*.1);k.copy_(torch.randn_like(k)*.08);v.copy_(torch.randn_like(v)*.2)
            beta.copy_(torch.rand_like(beta));decay.copy_(torch.rand_like(decay)*.15+.8)
            graph.replay();expected=baseline(ref);torch.npu.synchronize()
            if not bits_equal(s,ref) or not bits_equal(out,expected):
                exact=False;item.update(failed_step=i,max_state_abs=(s-ref).abs().max().item(),max_output_abs=(out-expected).abs().max().item());break
        item['recurrent_exact_256']=exact
        if exact:
            for gate in (0.0,1.0):
                s.normal_();ref.copy_(s);q.fill_(-.125);k.fill_(-.0625);v.zero_();beta.zero_();decay.fill_(gate)
                graph.replay();expected=baseline(ref);torch.npu.synchronize()
                assert bits_equal(s,ref) and bits_equal(out,expected),('signed-zero',blocks,gate)
            item['signed_zero_exact']=True
        del graph;gc.collect();torch.npu.empty_cache()
        if exact:
            q.copy_(torch.randn_like(q)*.1);k.copy_(torch.randn_like(k)*.08);v.copy_(torch.randn_like(v)*.2)
            beta.copy_(torch.rand_like(beta));decay.fill_(.9)
            for name,fn in (('torch',baseline),('direct',lambda:fused(blocks))):
                s.copy_(initial);graph=torch.npu.NPUGraph();stream=torch.npu.Stream()
                with torch.npu.stream(stream):
                    with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):
                        for _ in range(36):result=fn()
                torch.npu.synchronize();samples=[]
                for _ in range(14):
                    begin=time.perf_counter();graph.replay();torch.npu.synchronize();samples.append((time.perf_counter()-begin)*1000)
                item[name+'_36_calls_ms']=statistics.median(samples[2:]);del graph,result;gc.collect();torch.npu.empty_cache()
        records.append(item);print('DELTA_DIRECT',json.dumps(item),flush=True)
    Path(os.environ.get('FNX_DELTA_PROBE_OUTPUT','/tmp/fnx-delta-probe.json')).write_text(json.dumps(records,indent=2));print('R108_PROBE_COMPLETE',flush=True)

if __name__=='__main__':main()
