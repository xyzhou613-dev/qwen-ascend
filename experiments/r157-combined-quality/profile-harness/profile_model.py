"""One production-config Engine; same frozen HTTP prompts, sequential token positions."""
import hashlib,json,os,statistics,subprocess,time
from pathlib import Path
import torch,torch_npu
import profile_support as ps
P=Path(__file__).resolve().parent
def no_other_engine():
 active=[]
 for p in Path('/proc').iterdir():
  if not p.name.isdigit():continue
  try:
   cmd=p.joinpath('cmdline').read_bytes().split(b'\0')
   if b'fnx.server' in cmd or b'fnx.pipeline_worker' in cmd:active.append(p.name)
  except (FileNotFoundError,PermissionError,ProcessLookupError):pass
 assert not active,('REFUSED_SECOND_ENGINE',active)
def periods(rows):return [(b['end_mono_ns']-a['end_mono_ns'])/1e6 for a,b in zip(rows,rows[1:])]
@torch.no_grad()
def main():
 no_other_engine()
 from fnx.server import Engine
 e=Engine();m=e.m;records=json.loads((P/'runs.json').read_text()) if os.environ.get('R145_CONTEXTS')=='256k' else [];alignment=json.loads((P/'clock-alignment.json').read_text()) if os.environ.get('R145_CONTEXTS')=='256k' else []
 try:
  for context,target in [('8k',8064),('256k',262016)]:
   if context not in os.environ.get('R145_CONTEXTS','8k,256k').split(','):continue
   payload=json.loads((P/f'payload{context}.json').read_text())
   text=e.tok.apply_chat_template(payload['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
   ids=e.tok(text).input_ids;assert len(ids)==target,(context,len(ids))
   alignment.append(dict(context=context,samples=[]))
   for _ in range(12):
    a=time.time_ns();am=time.perf_counter_ns();cw,cm=m._rpc('clock');bm=time.perf_counter_ns();b=time.time_ns()
    alignment[-1]['samples'].append(dict(parent_start_ns=a,parent_end_ns=b,child_ns=cw,rtt_ns=bm-am,child_mono_ns=cm,parent_mono_mid_ns=(am+bm)//2))
   gen=e.generate(ids,124,temperature=0.,tpl_kw={'enable_thinking':False});tokens=[];sequence=0
   def step():
    nonlocal sequence
    ps.set_seq(sequence)
    a=time.time_ns();am=time.perf_counter_ns()
    while True:
     event=next(gen)
     if event[0]=='t':break
    b=time.time_ns();bm=time.perf_counter_ns();tokens.append(int(event[1]))
    row=dict(seq=sequence,start_ns=a,end_ns=b,start_mono_ns=am,end_mono_ns=bm,position=m.stage0._host_pos)
    sequence+=1;return row
   for _ in range(4):step()
   plan=[('baseline_before',None,16),('timing','AiCoreNone',24),('pipe','PipeUtilization',8),('memory','Memory',8),('access','MemoryAccess',8),('l2','L2Cache',8),('hardware',None,32),('baseline_after',None,16)]
   for label,metric,n in plan:
    tag=f'{context}_{label}';prof=None;system=None
    if metric:
     m._rpc('prof_start',str(P/'traces'/f'{tag}_stage1'),metric,tag)
     prof=ps.make_profiler(str(P/'traces'/f'{tag}_stage0'),metric);prof.start();ps.begin(tag,True)
    elif label=='hardware':
     path=P/'system'/context;path.mkdir(parents=True,exist_ok=True)
     log=(P/f'system-{context}.log').open('w')
     system=subprocess.Popen(['/usr/local/Ascend/ascend-toolkit/latest/bin/msprof',f'--output={path}','--sys-period=6','--sys-devices=all','--ai-core=off','--sys-hardware-mem=on','--sys-hardware-mem-freq=100','--sys-interconnection-profiling=on','--sys-interconnection-freq=50'],stdout=log,stderr=subprocess.STDOUT)
     time.sleep(3)
     ps.begin(tag,False);m._rpc('host_start',tag)
    rows=[step() for _ in range(n)]
    if metric:
     ps.finish(P/f'{tag}_host0.json');prof.stop();m._rpc('prof_stop',str(P/f'{tag}_host1.json'))
    elif label=='hardware':
     ps.finish(P/f'{tag}_host0.json');m._rpc('host_stop',str(P/f'{tag}_host1.json'))
     system.wait(timeout=90);log.close()
    p=periods(rows)
    record=dict(tag=tag,context=context,metric=metric,steps=n,rows=rows,mean_ms=statistics.mean(p),median_ms=statistics.median(p),periods_ms=p,system_returncode=system.returncode if system else None)
    records.append(record);(P/'runs.json').write_text(json.dumps(records,indent=2));(P/'clock-alignment.json').write_text(json.dumps(alignment,indent=2))
    print('PROFILE_WINDOW',tag,round(statistics.mean(p),4),round(statistics.median(p),4),rows[0]['position'],rows[-1]['position'],flush=True)
   assert len(tokens)==124 and m.stage0._host_pos<=m.P
   try:next(gen)
   except StopIteration:pass
   answer=e.tok.decode(tokens,skip_special_tokens=True)
   ref=None
   quality=dict(context=context,prompt_tokens=len(ids),generated_tokens=len(tokens),input_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),answer=answer,reference_starts_with_output=None)
   (P/f'{context}-output.json').write_text(json.dumps(quality,ensure_ascii=False,indent=2));print('OUTPUT_MATCH',context,quality['reference_starts_with_output'],flush=True)
  print('R145_PROFILE_COMPLETE',flush=True)
 finally:m.close()
if __name__=='__main__':main()
