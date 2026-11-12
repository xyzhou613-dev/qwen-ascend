"""All linear-layer AB projections, true single-token shapes and varied scales."""
import json
from pathlib import Path
import torch,torch_npu
from fnx import engine as E
from fnx.load import SRCDIR
from probe import weight,compare
P=Path(__file__).resolve().parent
@torch.no_grad()
def main():
 torch.npu.set_device(0);torch.manual_seed(1444);rows=[]
 types=json.load(open(Path(SRCDIR)/'config.json'))['text_config']['layer_types']
 for i,t in enumerate(types):
  if t in E.FULL_TYPES:continue
  ws=[weight(i,'linear_attn.in_proj_'+n) for n in ('b','a')];cat=torch.cat(ws,0)
  a=[];b=[]
  for j in range(128):
   x=torch.randn((1,1,E.H),device='npu:0',dtype=torch.bfloat16)*(.001,.03,.3,3,30)[j%5]
   a.append(torch.cat([E.lin(x,w) for w in ws],-1));b.append(E.lin(x,cat))
  torch.npu.synchronize();r=compare(torch.cat(a,0),torch.cat(b,0));rows.append(dict(layer=i,samples=128,comparison=r));print('RAW_AB_SINGLE_TOKEN',i,r,flush=True)
  del a,b,ws,cat
 P.joinpath('ab-probe.json').write_text(json.dumps(rows,indent=2));print('RAW_AB_PROBE_COMPLETE',all(r['comparison']['exact'] for r in rows),flush=True)
if __name__=='__main__':main()
