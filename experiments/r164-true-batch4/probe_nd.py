"""2.375 GiB B4/256k caches under the existing 1/24 allocator limit."""
import json,os,sys,time,traceback,urllib.request
from pathlib import Path
import torch,torch_npu
import write_nd as W
P=Path(__file__).resolve().parent
torch.set_num_threads(1);torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(1/24,0)
@torch.no_grad()
def run():
    h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
    assert h['status']=='ok' and h['worker_alive'] and h['service_revision']=='r152'
    baseline=h['pipeline_transfers']
    assert torch.npu.mem_get_info()[0]>4*1024**3
    # Small dense references ensure exactly the same update semantics.
    for dtype in (torch.bfloat16,torch.float32):
        p=torch.tensor([3,7,13,19],device='npu',dtype=torch.int64)
        var=torch.zeros(4,32,8,dtype=dtype,device='npu');ref=var.clone()
        update=torch.randn(4,8,dtype=dtype,device='npu')
        ref.scatter_(1,p[:,None,None].expand(4,1,8),update[:,None,:]);W.index(var,update,p)
        assert torch.equal(var,ref)
    var=torch.zeros(4,2,32,256,dtype=torch.bfloat16,device='npu');ref=var.clone()
    update=torch.randn(4,2,256,dtype=torch.bfloat16,device='npu')
    ref.scatter_(2,p[:,None,None,None].expand(4,2,1,256),update[:,:,None,:]);W.kv(var,update,p)
    assert torch.equal(var,ref)
    del var,ref,update;torch.npu.empty_cache()
    k=torch.zeros(4,2,262144,256,dtype=torch.bfloat16,device='npu');v=torch.zeros_like(k)
    ik=torch.zeros(4,262144,128,dtype=torch.bfloat16,device='npu')
    keys=torch.zeros(4,65536,128,dtype=torch.float32,device='npu')
    ku=torch.randn(4,2,256,dtype=torch.bfloat16,device='npu');vu=torch.randn_like(ku)
    iu=torch.randn(4,128,dtype=torch.bfloat16,device='npu');gu=torch.randn(4,128,dtype=torch.float32,device='npu')
    p=torch.tensor([3,511,4093,262140],device='npu',dtype=torch.int64)
    def step():
        W.kv(k,ku,p);W.kv(v,vu,p);W.index(ik,iu,p);W.index(keys,gu,p//4)
    def check():
        for var,u,positions in ((k,ku,p),(v,vu,p)):
            got=var.gather(2,positions[:,None,None,None].expand(4,2,1,256)).squeeze(2)
            assert torch.equal(got,u)
            assert not bool(var[:,:,0,:].count_nonzero())
        for var,u,positions in ((ik,iu,p),(keys,gu,p//4)):
            assert torch.equal(var.gather(1,positions[:,None,None].expand(4,1,128)).squeeze(1),u)
    before=torch.npu.memory_allocated();stream=torch.npu.Stream()
    with torch.npu.stream(stream):
        step();stream.synchronize();check();graph=torch.npu.NPUGraph()
        with torch.npu.graph(graph,stream=stream):step()
        stream.synchronize();check()
        p.copy_(torch.tensor([7,515,4097,262141],device='npu',dtype=torch.int64))
        ku.normal_();vu.normal_();iu.normal_();gu.normal_();graph.replay();stream.synchronize();check()
    peak=torch.npu.max_memory_allocated();allocated=torch.npu.memory_allocated()
    h2=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
    assert baseline==h2['pipeline_transfers'] and h2['worker_alive']
    assert torch.npu.mem_get_info()[0]>2*1024**3
    assert peak-before<64*1024**2,('LARGE_WRITE_WORKSPACE',peak-before)
    return dict(status='passed',small_scatter_reference_exact=True,fresh_graph_writes_exact=True,
        cache_gib=2.375,allocated_gib=allocated/2**30,peak_gib=peak/2**30,
        additional_peak_mib=(peak-before)/2**20,production_transfers_unchanged=baseline)
try:
    r=run();(P/'probe-nd.json').write_text(json.dumps(r,indent=2));print(json.dumps(r),flush=True);os._exit(0)
except Exception as ex:
    (P/'probe-nd.json').write_text(json.dumps(dict(status='failed',error=repr(ex)),indent=2))
    traceback.print_exc();sys.stderr.flush();os._exit(1)
