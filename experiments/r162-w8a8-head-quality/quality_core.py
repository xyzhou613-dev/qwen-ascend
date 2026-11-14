"""Frozen-ID scoring on the existing Engine; never loads another model."""
def score_ids(model,ids,scored,mode='decode',legacy=False):
 import torch
 assert 0<scored<len(ids) and mode in ('decode','prefill')
 warm=len(ids)-1-scored;losses=[];argmax=[]
 model.reset()
 try:
  if mode=='prefill' and warm:
   model.plan_prefill(warm);model.prefill_chunk(ids[:warm]);model.finish_prefill()
  for i in range(warm if mode=='prefill' else 0,len(ids)-1):
   if i<warm:model.step_pad(int(ids[i]),False);continue
   if legacy:
    model.step_pad(int(ids[i]),True);lg=model.row_logits().float().view(-1)
    losses.append(float(-torch.log_softmax(lg,-1)[int(ids[i+1])]))
    argmax.append(int(lg.argmax()))
   else:
    value=model.step_score(int(ids[i]),int(ids[i+1]));losses.append(value[0]);argmax.append(value[1])
  assert len(losses)==scored
  return dict(mean_nll=sum(losses)/len(losses),nll_tokens=losses,argmax=argmax,scored=len(losses),warm=warm,mode=mode,legacy=legacy)
 finally:model.reset()
