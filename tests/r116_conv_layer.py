"""Compare compact native DeltaNet with the reference on a real layer."""
import gc,json,os,statistics
from pathlib import Path
import torch,torch_npu
from fnx import engine as E
from fnx.load import load_layer,QDIR

def equal(a,b):return torch.equal(a.view(torch.int32 if a.dtype==torch.float32 else torch.int16),b.view(torch.int32 if b.dtype==torch.float32 else torch.int16))

@torch.no_grad()
def main():
    torch.npu.set_device(0);torch.manual_seed(109109)
    W=load_layer(os.path.join(QDIR,'L00.safetensors'),0,'npu:0');L=E.FnLayer(0,'linear',W,'npu:0');del W
    h=torch.randn(1,1,E.H,dtype=torch.bfloat16,device='npu:0')*.03
    os.environ['FNX_DN_NATIVE']='1';os.environ['FNX_DN_COMPACT']='1';os.environ['FNX_CONV_NATIVE']='1'
    L._deltanet(h);torch.npu.synchronize()
    graph=torch.npu.NPUGraph();stream=torch.npu.Stream()
    with torch.npu.stream(stream):
        with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):actual=L._deltanet(h)
    torch.npu.synchronize();L.S.zero_();L.conv.zero_()
    for i in range(256):
        h.copy_(torch.randn_like(h)*(.03 if i%2 else .1));before=L.S.clone();conv=L.conv.clone();torch.npu.synchronize()
        graph.replay();torch.npu.synchronize();saved=actual.clone();after=L.S.clone();after_conv=L.conv.clone()
        L.S.copy_(before);L.conv.copy_(conv);os.environ['FNX_CONV_NATIVE']='0'
        expected=L._deltanet(h);torch.npu.synchronize()
        assert equal(saved,expected),('output',i,(saved.float()-expected.float()).abs().max().item())
        assert equal(after,L.S),('state',i,(after-L.S).abs().max().item())
        assert equal(after_conv,L.conv),('conv',i)
        os.environ['FNX_CONV_NATIVE']='1'
        if i%64==63:print('REAL_DELTA_EXACT',i+1,flush=True)
    del graph, actual
    gc.collect()
    timing={}
    for name,compact in (('baseline','0'),('compact','1'),('baseline_repeat','0')):
        os.environ['FNX_CONV_NATIVE']=compact
        L.S.zero_();L.conv.zero_();torch.npu.synchronize()
        with torch.npu.stream(stream):
            graph=torch.npu.NPUGraph()
            with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):
                output=L._deltanet(h)
        graph.replay();torch.npu.synchronize()
        samples=[]
        for _ in range(5):
            start=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True)
            start.record()
            for _ in range(128):graph.replay()
            end.record();end.synchronize()
            samples.append(start.elapsed_time(end)*1000/128)
        timing[name]=statistics.median(samples)
        del graph, output
        gc.collect()
    result={'layer':0,'steps':256,'graph_dynamic_input':True,'state_fp32_bit_exact':True,'conv_bf16_bit_exact':True,'output_bf16_bit_exact':True,'layer_graph_us':timing}
    print('DELTA_LAYER_US',json.dumps(timing),flush=True)
    Path(os.environ.get('FNX_DELTA_LAYER_OUTPUT','/tmp/fnx-delta-layer.json')).write_text(json.dumps(result,indent=2));print('R116_CONV_LAYER_ACCEPT',flush=True)

if __name__=='__main__':main()
