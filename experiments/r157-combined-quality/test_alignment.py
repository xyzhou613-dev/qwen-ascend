from quality_core import score_ids
class Model:
 def __init__(self,fail=False):self.events=[];self.fail=fail
 def reset(self):self.events.append(('reset',))
 def plan_prefill(self,n):self.events.append(('plan',n))
 def prefill_chunk(self,ids):self.events.append(('prefill',ids))
 def finish_prefill(self):self.events.append(('finish',))
 def step_pad(self,tok,head):self.events.append(('warm',tok,head))
 def step_score(self,tok,target):
  self.events.append(('score',tok,target))
  if self.fail:raise RuntimeError('injected scoring failure')
  return float(tok+target),target
for mode in ('decode','prefill'):
 m=Model();r=score_ids(m,[10,20,30,40,50],2,mode)
 assert r['scored']==2 and r['mean_nll']==80 and r['argmax']==[40,50]
 assert [x for x in m.events if x[0]=='score']==[('score',30,40),('score',40,50)]
 assert m.events[0]==m.events[-1]==('reset',)
 if mode=='prefill':assert ('prefill',[10,20]) in m.events
 else:assert [x for x in m.events if x[0]=='warm']==[('warm',10,False),('warm',20,False)]
m=Model(True)
try:score_ids(m,[10,20,30],1)
except RuntimeError:assert m.events[-1]==('reset',)
else:raise AssertionError('failure should propagate')
print('SCORING_ALIGNMENT_AND_CLEANUP_PASS')
