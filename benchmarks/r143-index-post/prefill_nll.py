"""Score a frozen token document through the real production prefill policy.

Run only in a guarded exclusive diagnostic environment with the complete
production settings. A second Engine must not be started beside the live one.
The alignment intentionally matches r142: a 200-token tail has199 scored targets.
"""
import argparse,hashlib,json
from pathlib import Path
import torch
@torch.no_grad()
def score_model(model,ids,tail=200):
 assert len(ids)>tail>=2
 pre=len(ids)-tail;model.reset();model.plan_prefill(pre)
 model.prefill_chunk(ids[:pre]);model.finish_prefill();torch.npu.synchronize()
 losses=[];argmax=[];hits=0
 try:
  for i in range(pre,len(ids)-1):
   model.step_pad(int(ids[i]),True)
   logits=model.row_logits().float().view(-1);lp=torch.log_softmax(logits,-1);am=int(logits.argmax());target=int(ids[i+1])
   losses.append(float(-lp[target]));argmax.append(am);hits+=int(am==target)
  n=len(losses)
  return dict(tokens=len(ids),scored=n,mean_nll=round(sum(losses)/n,6),top1=round(hits/n,4),
      token_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),nll_tokens=losses,argmax=argmax)
 finally:model.reset()
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ids',type=Path,default=Path(__file__).with_name('quality-input-ids.json'));p.add_argument('--output',type=Path)
 args=p.parse_args();ids=json.loads(args.ids.read_text());assert len(ids)==6000
 from fnx.server import Engine
 e=Engine()
 try:
  result=score_model(e.m,ids)
  if args.output:args.output.write_text(json.dumps(result,indent=2))
  print('REAL_PREFILL_NLL',json.dumps(result),flush=True)
 finally:e.m.close()
