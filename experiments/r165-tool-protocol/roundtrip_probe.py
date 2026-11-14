"""Assemble a real streamed call by index and return its typed tool result."""
import hashlib,json,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
j=json.loads((P/'live-acceptance.json').read_text());r=next(r for r in j['records'] if r['name']=='typed_sse_usage')
assembled={}
for event in r['events']:
    for ch in event.get('choices',[]):
        for delta in ch.get('delta',{}).get('tool_calls',[]):
            i=delta['index'];target=assembled.setdefault(i,dict(id='',type='function',function=dict(name='',arguments='')))
            if delta.get('id'):target['id']=delta['id']
            for key in ('name','arguments'):target['function'][key]+=delta.get('function',{}).get(key,'')
calls=[assembled[i] for i in sorted(assembled)];assert len(calls)==1
args=json.loads(calls[0]['function']['arguments'])
assert type(args['count']) is int and type(args['enabled']) is bool and isinstance(args['tags'],list) and isinstance(args['options'],dict)
messages=r['request']['messages']+[dict(role='assistant',content=None,tool_calls=calls),
    dict(role='tool',tool_call_id=calls[0]['id'],content=json.dumps(dict(ok=True,received=args),ensure_ascii=False)),
    dict(role='user',content='以上工具结果已经确认。请只回复协议回传成功，不要再调用工具。')]
body=dict(model='qwen3.8-flash-next-w4',messages=messages,tools=r['request']['tools'],tool_choice='none',max_completion_tokens=64,temperature=0)
start=time.perf_counter();req=urllib.request.Request('http://127.0.0.1:9002/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json','Connection':'close'})
with urllib.request.urlopen(req,timeout=60) as response:result=json.load(response)
message=result['choices'][0]['message'];assert not message.get('tool_calls') and '协议回传成功' in message['content']
out=dict(status='passed',elapsed_s=time.perf_counter()-start,streamed_call_assembled_by_index=True,typed_arguments_preserved=True,
    assistant_null_content_accepted=True,tool_result_roundtrip=True,real_external_tool_executed=False,request=body,response=result)
(P/'roundtrip-probe.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print('LIVE_TYPED_TOOL_RESULT_ROUNDTRIP_PASS',round(out['elapsed_s'],3))
