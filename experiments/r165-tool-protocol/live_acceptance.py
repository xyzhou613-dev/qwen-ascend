"""Read-only live protocol requests; returned fake tool calls are not executed."""
import hashlib,json,re,subprocess,time,urllib.error,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;BASE='http://127.0.0.1:9002'
def get(route):
    with urllib.request.urlopen(urllib.request.Request(BASE+route,headers={'Connection':'close'}),timeout=5) as r:return r.read()
def health():
    h=json.loads(get('/health'));assert h['status']=='ok' and h['service_revision']=='r152' and h['worker_alive'] and h.get('tool_protocol_revision')=='r165';return h
def count():return float(re.search(r'^fnx_requests_total ([0-9.]+)$',get('/metrics').decode(),re.M).group(1))
def idle():
    a=count();h=health();time.sleep(.4);b=count();h2=health()
    sockets=subprocess.check_output(['ss','-tnH','state','established','sport','=',':9002'],text=True).strip()
    return a==b and h['pipeline_transfers']==h2['pipeline_transfers'] and not sockets
def procs():
    out=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            argv=p.joinpath('cmdline').read_bytes().split(b'\0')
            if b'fnx.server' in argv or b'fnx.pipeline_worker' in argv:
                out.append(dict(pid=int(p.name),birth=p.joinpath('stat').read_text().split()[21]))
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return sorted(out,key=lambda x:x['pid'])
def tool(name,properties):return dict(type='function',function=dict(name=name,description='只回显诊断参数，不执行任何操作。',parameters=dict(type='object',properties=properties,required=list(properties))))
types=tool('audit_types',dict(count={'type':'integer'},enabled={'type':'boolean'},tags={'type':'array','items':{'type':'string'}},options={'type':'object'},label={'type':'string'}))
echo=tool('mcp__audit__echo',dict(text={'type':'string'}))
plan=tool('audit_plan',dict(todos={'type':'array','items':{'type':'object','properties':{'content':{'type':'string'},'status':{'type':'string'}},'required':['content','status']}}))
plan_text='请使用audit_plan回显以下六项待办，不执行任务，每项status为pending，content保留完整句子：1.确认两张NPU的型号、设备编号和PCI地址。2.检查推理服务版本、启动参数与模型权重目录。3.核对工具声明、参数schema和流式接口字段。4.分别验证整数、布尔、数组和对象参数的类型。5.检查长参数是否完整以及停止原因是否正确。6.汇总所有检查结果并保存详细的诊断报告。只输出工具调用。'
cases=[
 ('typed_nonstream','请调用audit_types：count为整数3，enabled为布尔true，tags为数组["甲","乙"]，options为对象{"mode":"fast"}，label为字符串123。只输出调用。',dict(tools=[types],max_completion_tokens=512)),
 ('typed_sse_usage','请调用audit_types：count为整数3，enabled为布尔true，tags为数组["甲","乙"]，options为对象{"mode":"fast"}，label为字符串123。只输出调用。',dict(tools=[types],stream=True,stream_options={'include_usage':True},max_completion_tokens=512)),
 ('required_single','请把这段文字交给回声工具原样返回：你好，世界😀',dict(tools=[echo],tool_choice='required',stream=True,stream_options={'include_usage':True},max_completion_tokens=256)),
 ('required_multi','请把文字你好交给回声工具回显。',dict(tools=[types,echo],tool_choice='required',max_completion_tokens=256)),
 ('specific_echo','请原样回显这段文字：你好，世界😀',dict(tools=[types,echo],tool_choice={'type':'function','function':{'name':'mcp__audit__echo'}},max_completion_tokens=256)),
 ('none_policy','请调用mcp__audit__echo，text参数为诊断回声。只输出调用。',dict(tools=[echo],tool_choice='none',max_completion_tokens=128)),
 ('unknown_named_choice','只回复OK。',dict(tools=[echo],tool_choice={'type':'function','function':{'name':'does_not_exist'}},max_completion_tokens=32)),
 ('invalid_choice','只回复OK。',dict(tools=[echo],tool_choice='not_a_choice',max_completion_tokens=32)),
 ('long_modern_limit',plan_text,dict(tools=[plan],max_completion_tokens=1024)),
 ('both_limits',plan_text,dict(tools=[plan],max_tokens=32,max_completion_tokens=1024)),
 ('null_legacy_limit',plan_text,dict(tools=[plan],max_tokens=None,max_completion_tokens=1024)),
 ('truncated_sse',plan_text,dict(tools=[plan],stream=True,max_completion_tokens=32))]
initial=health();original=procs();source=Path('/root/ascend-dev/fnx/fnx/server.py');validation=source.with_name('request_validation.py')
hashes={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in (source,validation,source.with_name('tool_protocol.py'))};records=[]
(P/'live-before.json').write_text(json.dumps(dict(health=initial,processes=original,hashes=hashes),indent=2))
for name,prompt,extra in cases:
    body=dict(model='qwen3.8-flash-next-w4',messages=[dict(role='user',content=prompt)],temperature=0,**extra)
    before=count();start=time.perf_counter();events=[];done=False
    req=urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json','Connection':'close'})
    try:
        with urllib.request.urlopen(req,timeout=90) as response:
            status=response.status
            if extra.get('stream'):
                for line in response:
                    if not line.startswith(b'data: '):continue
                    raw=line[6:].strip()
                    if raw==b'[DONE]':done=True;break
                    events.append(json.loads(raw))
                raw_response=None
            else:raw_response=json.load(response)
    except urllib.error.HTTPError as error:status=error.code;raw_response=json.loads(error.read())
    row=dict(name=name,status='completed',http_status=status,wall_s=time.perf_counter()-start,request=body,
             response=raw_response,events=events,done=done,request_counter_delta=count()-before)
    if events:
        calls=[c for e in events for ch in e.get('choices',[]) for c in ch.get('delta',{}).get('tool_calls',[])]
        row['summary']=dict(calls=calls,call_indices=[c.get('index') for c in calls],
            content=''.join(ch.get('delta',{}).get('content','') for e in events for ch in e.get('choices',[])),
            finish_reason=next((ch['finish_reason'] for e in reversed(events) for ch in e.get('choices',[]) if ch.get('finish_reason')),None),
            usage_chunk=any(e.get('choices')==[] and 'usage' in e for e in events),
            output_tokens=next((e['fnx_stats']['tokens'] for e in reversed(events) if 'fnx_stats' in e),None))
    elif status==200:
        ch=raw_response['choices'][0];row['summary']=dict(calls=ch['message'].get('tool_calls',[]),content=ch['message'].get('content'),
            finish_reason=ch['finish_reason'],output_tokens=raw_response['usage']['completion_tokens'])
    records.append(row);(P/'live-acceptance.json').write_text(json.dumps(dict(status='running',records=records),ensure_ascii=False,indent=2))
    summary=row.get('summary',{});print('LIVE_COMPLETE',name,status,round(row['wall_s'],2),summary.get('output_tokens'),summary.get('finish_reason'),flush=True)
    if name in ('unknown_named_choice','invalid_choice'):
        assert status==400,(name,'expected400',status)
        continue
    assert status==200,(name,status)
    cs=summary['calls']
    if name in ('typed_nonstream','typed_sse_usage'):
        assert cs and summary['finish_reason']=='tool_calls'
        args=json.loads(cs[0]['function']['arguments'])
        assert {k:type(v).__name__ for k,v in args.items()}==dict(count='int',enabled='bool',tags='list',options='dict',label='str'),(name,args)
    if name in ('required_single','required_multi','specific_echo'):
        assert cs and summary['finish_reason']=='tool_calls',(name,summary)
        assert all(c['function']['name'] in [t['function']['name'] for t in extra['tools']] for c in cs)
        if name!='required_multi':
            assert all(c['function']['name']=='mcp__audit__echo' for c in cs)
            assert json.loads(cs[0]['function']['arguments'])['text']=='你好，世界😀'
    if extra.get('stream'):
        assert done
        assert [c['index'] for c in cs]==list(range(len(cs)))
        if extra.get('stream_options',{}).get('include_usage'):
            usage=[e['usage'] for e in events if e.get('choices')==[] and 'usage' in e]
            assert len(usage)==1 and usage[0]['completion_tokens']==summary['output_tokens']
            assert usage[0]['total_tokens']==usage[0]['prompt_tokens']+usage[0]['completion_tokens']
    if name=='none_policy':assert not cs and summary['finish_reason']!='tool_calls'
    if name in ('long_modern_limit','both_limits','null_legacy_limit'):
        assert summary['output_tokens']>128 and cs,(name,summary)
        todos=json.loads(cs[0]['function']['arguments'])['todos'];assert isinstance(todos,list) and len(todos)==6
    if name=='truncated_sse':assert summary['finish_reason']=='length' and not cs and summary['output_tokens']==32
assert procs()==original,'PRODUCTION_PROCESS_CHANGED'
assert all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in hashes.items())
final=health()
result=dict(status='passed',scope='r152 engine with r165 protocol live HTTP; no returned tool is executed',records=records,
    production_processes_unchanged=True,production_sources_unchanged=True,processes=original,source_hashes=hashes,final_health=final)
(P/'live-acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));print('LIVE_PROTOCOL_ACCEPTANCE_PASS',len(records),flush=True)
