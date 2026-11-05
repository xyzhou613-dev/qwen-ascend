"""Full L00 parity including recurrent state and dynamic expert routing."""
import gc,json,os,statistics,time
from pathlib import Path
import torch,torch_npu
from fnx import engine as E
from fnx.load import load_layer,QDIR

def flags(enabled):
    for key in ('FNX_MIX_REDUCE','FNX_MOE_REDUCE'):os.environ[key]=str(int(enabled))

def equal(a,b):
    dtype=torch.int32 if a.dtype==torch.float32 else torch.int16
    return torch.equal(a.view(dtype),b.view(dtype))

@torch.no_grad()
def main():
    torch.npu.set_device(0);torch.manual_seed(119119)
    W=load_layer(os.path.join(QDIR,'L00.safetensors'),0,'npu:0')
    layer=E.FnLayer(0,'linear',W,'npu:0');del W
    h=torch.randn(1,1,E.HCH,dtype=torch.bfloat16,device='npu:0')*.03
    pos=torch.zeros(1,dtype=torch.int64,device='npu:0')
    flags(True);layer.forward(h,pos);torch.npu.synchronize()
    stream=torch.npu.Stream();graph=torch.npu.NPUGraph()
    with torch.npu.stream(stream):
        with torch.npu.graph(graph,stream=stream,capture_error_mode='relaxed'):
            actual=layer.forward(h,pos)
    torch.npu.synchronize();layer.S.zero_();layer.conv.zero_()
    for i in range(128):
        h.copy_(torch.randn_like(h)*(.03 if i%2 else .1));pos.fill_(i)
        before=layer.S.clone();conv=layer.conv.clone();torch.npu.synchronize()
        graph.replay();torch.npu.synchronize()
        saved=actual.clone();after=layer.S.clone();after_conv=layer.conv.clone()
        layer.S.copy_(before);layer.conv.copy_(conv);flags(False)
        expected=layer.forward(h,pos);torch.npu.synchronize()
        assert equal(saved,expected),('decode_output',i,(saved.float()-expected.float()).abs().max().item())
        assert equal(after,layer.S),('decode_state',i)
        assert equal(after_conv,layer.conv),('decode_conv',i)
        flags(True)
        if i%32==31:print('REAL_LAYER_DECODE_EXACT',i+1,flush=True)
    del graph,actual;gc.collect()
    chunks=[]
    for T in (128,512,8192):
        h=torch.randn(T,1,E.HCH,dtype=torch.bfloat16,device='npu:0')*.03
        layer.S.zero_();layer.conv.zero_();flags(False)
        expected=layer.forward_chunk(h,0,T);torch.npu.synchronize()
        saved=expected.clone();after=layer.S.clone();after_conv=layer.conv.clone()
        del expected
        layer.S.zero_();layer.conv.zero_();flags(True)
        actual=layer.forward_chunk(h,0,T);torch.npu.synchronize()
        assert equal(actual,saved),('chunk_output',T,(saved.float()-actual.float()).abs().max().item())
        assert equal(after,layer.S),('chunk_state',T)
        assert equal(after_conv,layer.conv),('chunk_conv',T)
        chunks.append(T);print('REAL_LAYER_CHUNK_EXACT',T,flush=True)
        del actual,saved,after,after_conv,h;gc.collect();torch.npu.empty_cache()
    result={'layer':0,'decode_steps':128,'graph_dynamic_input':True,'chunks':chunks,
            'state_fp32_bit_exact':True,'conv_bf16_bit_exact':True,'output_bf16_bit_exact':True}
    Path(os.environ.get('FNX_REDUCE_LAYER_OUTPUT','/tmp/fnx-reduce-layer.json')).write_text(json.dumps(result,indent=2))
    print('R119_REDUCE_LAYER_ACCEPT',flush=True)

if __name__=='__main__':main()
