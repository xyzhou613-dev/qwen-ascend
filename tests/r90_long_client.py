"""One exact near-capacity HTTP prompt with three distributed facts."""
import json,os,time,uuid,urllib.request,urllib.error
from pathlib import Path
from transformers import AutoTokenizer

P=int(os.environ['TEST_CTX']);target=P-128
max_output=int(os.environ.get('LONG_OUTPUT','64'))
tk=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next')
text=Path('/root/ascend-dev/eval_text.txt').read_text(encoding='utf-8')
unit=tk(text+'\n').input_ids
filler=(unit*(P//len(unit)+2))[:P]
nonce=uuid.uuid4().hex
facts=('项目海棠的验收码是7319。','项目松柏的验收码是8652。','项目银杏的验收码是2946。')
n=target-128
for _ in range(15):
    points=[n//10,n//2,n*9//10]
    parts=['本次文档编号：'+nonce+'。\n'];begin=0
    for point,fact in zip(points,facts):
        parts.extend([tk.decode(filler[begin:point]),'\n'+fact+'\n']);begin=point
    question='\n资料结束。请从资料提取海棠、松柏、银杏三个项目的验收码，按此顺序回答项目名与验收码。'
    if os.environ.get('LONG_EXPLAIN')=='1':
        question+='列完三个验收码后，再详细介绍资料中涉及的推理服务和长文档处理技术，连续写不少于800字。'
    parts.extend([tk.decode(filler[begin:n]),question])
    messages=[{'role':'user','content':''.join(parts)}]
    template=tk.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    ids=tk(template).input_ids
    if len(ids)==target:break
    n+=target-len(ids)
assert len(ids)==target,len(ids)
base='http://127.0.0.1:8000'
def post(path,data):
    req=urllib.request.Request(base+path,data=json.dumps(data).encode(),headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=1800) as r:return json.load(r)
quality=post('/fnx/loss',{'text':text,'max_tokens':200})
print('QUALITY',P,{k:quality.get(k) for k in ('mean_nll','ppl','top1_acc','n')},flush=True)
assert abs(quality['mean_nll']-2.164792)<.01
payload={'messages':messages,'max_tokens':max_output,'temperature':0,'enable_thinking':False,'stream':True}
req=urllib.request.Request(base+'/v1/chat/completions',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
start=time.perf_counter();first=None;answer=[];stats=None
with urllib.request.urlopen(req,timeout=1800) as response:
    for line in response:
        if line.startswith(b': prefill '):
            print('PROGRESS',P,line.decode().strip(),'elapsed',round(time.perf_counter()-start,2),flush=True)
        if not line.startswith(b'data: '):continue
        body=line[6:].strip()
        if body==b'[DONE]':break
        event=json.loads(body)
        if 'fnx_stats'in event:stats=event['fnx_stats']
        content=event['choices'][0]['delta'].get('content','')
        if content:
            if first is None:first=time.perf_counter()-start
            answer.append(content)
elapsed=time.perf_counter()-start
answer=''.join(answer)
result={'ctx':P,'prompt_tokens':target,'ttft_s':round(first,3),'e2e_s':round(elapsed,3),'stats':stats,
        'answer':answer,'recall_hits':sum(code in answer for code in ('7319','8652','2946'))}
out=Path(os.environ.get('RESULT_DIR','/root/ascend-dev/r52'));out.mkdir(exist_ok=True)
(out/f'result_{P}.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print('LONG_RESULT',json.dumps(result,ensure_ascii=False),flush=True)
assert stats['n_prefill']>=target-4
assert result['recall_hits']==3,answer
payload['stream']=False
start=time.perf_counter();repeat=post('/v1/chat/completions',payload)
print('REPEAT',P,'seconds',round(time.perf_counter()-start,3),'stats',repeat['fnx_stats'],'answer',repeat['choices'][0]['message']['content'],flush=True)
assert repeat['fnx_stats']['n_prefill']==0
assert all(code in repeat['choices'][0]['message']['content'] for code in ('7319','8652','2946'))
print('LONG_ACCEPT',P,flush=True)

if os.environ.get('LONG_CONTINUE')=='1':
    follow_messages=messages+[{'role':'assistant','content':'已收到资料。'},
                              {'role':'user','content':'海棠的验收码是多少？仅输出四位数字。'}]
    follow_text=tk.apply_chat_template(follow_messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    follow_ids=tk(follow_text).input_ids
    assert follow_ids[:len(ids)-1]==ids[:-1], 'continuation must match cached prefill'
    assert len(follow_ids)+32<=262144
    start=time.perf_counter()
    continued=post('/v1/chat/completions',{'messages':follow_messages,'max_tokens':32,'temperature':0,'enable_thinking':False})
    seconds=time.perf_counter()-start
    follow_answer=continued['choices'][0]['message']['content'];follow_stats=continued['fnx_stats']
    print('CONTINUATION',json.dumps({'seconds':seconds,'prompt_tokens':len(follow_ids),'answer':follow_answer,'stats':follow_stats},ensure_ascii=False),flush=True)
    assert '7319' in follow_answer
    assert 0<follow_stats['n_prefill']<128
    (out/'continuation.json').write_text(json.dumps({'seconds':seconds,'prompt_tokens':len(follow_ids),'answer':follow_answer,'stats':follow_stats},ensure_ascii=False,indent=2))
    print('CONTINUATION_ACCEPT',flush=True)
