"""Bounded traces from both r92 pipeline stages on real prompt histories.

Profiling intentionally drains at capture boundaries and disables conversation
snapshots. These traces are for attribution, never end-to-end benchmark claims.
"""
from pathlib import Path
import collections,csv,gc,json,os,time
import torch,torch_npu
from fnx.server import Engine

r=Path('/root/ascend-dev');out=r/'r93';traces=out/'traces';traces.mkdir(exist_ok=True)

def summarize():
    summaries={}
    for f in traces.rglob('*kernel_details.csv'):
        with f.open(encoding='utf-8-sig') as stream:
            rd=csv.DictReader(stream);cols=rd.fieldnames
            dur=next((x for x in cols if 'Duration' in x),None)
            typ=next((x for x in cols if x in ('Type','OP Type','Op Type','OP_Type')),None)
            if not dur or not typ:continue
            totals=collections.defaultdict(float);counts=collections.Counter()
            for row in rd:
                try:v=float(row[dur])
                except (ValueError,TypeError):continue
                op=row[typ]
                if not op or op=='N/A':op=row.get('Name','unknown')
                totals[op]+=v;counts[op]+=1
        total=sum(totals.values())
        key=str(f.relative_to(traces))
        summaries[key]={'device_total_ms':total/1000,'ops':[
            {'op':op,'ms':v/1000,'percent':100*v/max(total,1),'count':counts[op]}
            for op,v in sorted(totals.items(),key=lambda x:-x[1])[:20]]}
        print('PROFILE_SUMMARY',key,json.dumps(summaries[key]),flush=True)
    (out/'summaries.json').write_text(json.dumps(summaries,indent=2))

@torch.no_grad()
def main():
    e=Engine();m=e.m;events=[];active=None;profile=None;started=0;tagkind=None
    original_prefill=m.prefill_chunk;original_step=m.step_pad
    def begin(tag):
        nonlocal active,profile,started
        assert active is None
        m.finish_prefill();m._rpc('prof_start',str(traces/(tag+'_stage1')))
        profile=torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
            record_shapes=True,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(traces/(tag+'_stage0'))))
        profile.start();active=tag;started=time.perf_counter_ns()
    def finish():
        nonlocal active,profile
        m.finish_prefill();torch.npu.synchronize();ended=time.perf_counter_ns()
        profile.stop();m._rpc('prof_stop');events.append({'tag':active,'start_ns':started,'end_ns':ended,'wall_ms':(ended-started)/1e6})
        print('PROFILE_PHASE',events[-1],flush=True);active=None;profile=None
    phase='8k';long_chunks=0;decode_steps=0;prompt_length=0
    def prefill(ids):
        nonlocal long_chunks
        pos=m._host_pos
        if phase=='8k':begin('prefill8k')
        elif pos>=196608 and long_chunks<2:
            if long_chunks==0:begin('prefill256k_late')
            long_chunks+=1
        original_prefill(ids)
        if phase=='8k' or (active=='prefill256k_late' and long_chunks==2):finish()
    def step(tok,with_head=True):
        nonlocal decode_steps
        if with_head and decode_steps==0:begin('decode_'+phase)
        original_step(tok,with_head)
        if with_head:
            decode_steps+=1
            if decode_steps==16:finish()
    m.prefill_chunk=prefill;m.step_pad=step
    payload=json.loads((r/'r92/baseline8k/payload1.json').read_text())
    template=e.tok.apply_chat_template(payload['messages'],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=e.tok(template).input_ids
    for _ in e.generate(ids,32,0,{'enable_thinking':False}):pass
    phase='256k';decode_steps=0
    source=(r/'r90/r52_long_client.py').read_text().split("base='http://127.0.0.1:8000'")[0]
    namespace={};os.environ.update(TEST_CTX='262144',LONG_OUTPUT='128',LONG_EXPLAIN='1')
    exec(compile(source,'long-payload','exec'),namespace)
    ids=namespace['ids'];assert len(ids)==262016
    tokens=[]
    for item in e.generate(ids,128,0,{'enable_thinking':False}):
        if item[0]=='t':tokens.append(item[1])
    answer=e.tok.decode(tokens,skip_special_tokens=True)
    assert all(code in answer for code in ('7319','8652','2946')),answer
    (out/'events.json').write_text(json.dumps(events,indent=2))
    (out/'answer.txt').write_text(answer)
    m.close();summarize()
    print('R93_PROFILE_ACCEPT',flush=True)

if __name__=='__main__':main()
