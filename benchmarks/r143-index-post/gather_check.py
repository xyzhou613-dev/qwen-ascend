"""Recheck the deployed gather through its production wrapper with prepared inputs."""
from pathlib import Path
from types import SimpleNamespace
import json,torch,torch_npu
from fnx.expert_gather import gather
P=Path(__file__).resolve().parent
torch.npu.set_device(0);torch.manual_seed(1432);rows=[]
for k,g,n in ((2560,40,1280),(640,10,2560)):
 w=torch.randint(-2147483647,2147483647,(512,k,n//8),dtype=torch.int32,device='npu:0')
 s=torch.randn((512,g,n),dtype=torch.bfloat16,device='npu:0')*.01
 obj=SimpleNamespace(expert_major=True,_scales=s,w4=w,K=k,G=g,N=n,E=512)
 for case in range(3):
  ids=(torch.tensor([0,511,3,3,254,1,511,17,63,64],dtype=torch.int64,device=w.device) if case==0 else torch.randperm(512,device=w.device)[:10].to(torch.int64))
  rw=w.index_select(0,ids).permute(1,0,2).contiguous().view(k,-1)
  rs=s.index_select(0,ids).permute(1,0,2).contiguous().view(g,-1)
  for repeat in range(3):
   torch.npu.synchronize()
   a,b=gather(obj,ids);torch.npu.synchronize()
   row=dict(K=k,G=g,N=n,case=case,repeat=repeat,weights_exact=torch.equal(a,rw),scales_exact=torch.equal(b.view(torch.int16),rs.view(torch.int16)))
   rows.append(row);print(row,flush=True)
  del rw,rs,ids,a,b
 del obj,w,s
 torch.npu.empty_cache()
(P/'gather-check.json').write_text(json.dumps(rows,indent=2))
assert all(r['weights_exact'] and r['scales_exact'] for r in rows)
print('DEPLOYED_GATHER_VALID_WRAPPER_EXACT',flush=True)
