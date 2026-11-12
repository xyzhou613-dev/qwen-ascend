import importlib.util,json,os
from pathlib import Path
import torch,torch_npu
from fnx import engine as E
from probe import compare
P=Path(__file__).resolve().parent
from fnx import norm_post as m
os.environ['FNX_NORM_POST']='0'
with torch.no_grad():
 torch.npu.set_device(0)
 vals=[0.,-0.,1e-22,-1e-22,1e-20,-1e-20,1e-10,-1e-10,1e-4,-1e-4,1.,-1.,1e4,-1e4,1e19,-1e19,1e20,-1e20]
 rows=[]
 for v in vals:
  x=torch.full((1,16,128),v,device='npu:0',dtype=torch.bfloat16)
  a=E.l2norm(x);b=m.run(x);torch.npu.synchronize();r=compare(a,b);rows.append(dict(value=v,comparison=r));assert r['exact'],rows[-1]
 P.joinpath('norm-extremes.json').write_text(json.dumps(rows,indent=2));print('NORM_EXTREMES_BIT_EXACT',len(rows),flush=True)
