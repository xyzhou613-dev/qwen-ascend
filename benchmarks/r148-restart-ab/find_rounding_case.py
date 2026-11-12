"""Bounded operator-only search for rare normalization rounding differences."""
import gc,importlib.util,json,os,sys,time
from pathlib import Path
import torch,torch_npu
import torch.nn.functional as F
P=Path(__file__).resolve().parent;C=Path('/models/fnx-diagnostics/r146-norm-pair-20261112')
sys.path.insert(0,str(C))
from fnx import norm_post
spec=importlib.util.spec_from_file_location('probe_fused',C/'fused_norm_post.py');fused=importlib.util.module_from_spec(spec);spec.loader.exec_module(fused)
torch.set_num_threads(1);torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(1/128,0);torch.manual_seed(1483)
start=time.monotonic();checked=0;found=[]
for iteration in range(2048):
    if time.monotonic()-start>120:break
    assert torch.npu.mem_get_info()[0]>1536*1024**2,'online HBM headroom'
    amp=[.003,.03,.3,1.,3.,10.,.1,.01][iteration%8]
    x=torch.randn(64,16,128,dtype=torch.bfloat16,device='npu')*amp
    if iteration%2:x=F.silu(x)
    a=norm_post.run(x);b=fused.run(x,scale_query=False)
    mask=(a!=b).any(-1).cpu();checked+=64*16
    for batch,row in mask.nonzero().tolist()[:4]:
        # Recheck in the original single-token 16-row shape, not just the
        # batched search shape whose CANN reduction tiling may differ.
        probe=x[batch,row].view(1,1,128).expand(1,16,128).contiguous()
        ref=norm_post.run(probe);new=fused.run(probe,scale_query=False)
        aa=ref.cpu().view(torch.int16);bb=new.cpu().view(torch.int16)
        if not torch.equal(aa,bb):
            data=dict(input=probe.cpu(),reference=ref.cpu(),candidate=new.cpu(),iteration=iteration,amplitude=amp,silu=bool(iteration%2))
            torch.save(data,P/f'rounding-case-{len(found)}.pt');found.append(dict(iteration=iteration,amplitude=amp,silu=bool(iteration%2),different_values=int((aa!=bb).sum())))
    if found:break
    if iteration%64==0:print('ROUNDING_SEARCH',checked,flush=True)
    time.sleep(.01)
r=dict(checked_rows=checked,found=found,elapsed_s=time.monotonic()-start,max_allocated_bytes=torch.npu.max_memory_allocated(),scope='operator only; no Engine')
(P/'rounding-search.json').write_text(json.dumps(r,indent=2));print('ROUNDING_SEARCH_COMPLETE',r,flush=True)
