"""Actual request handler + real tokenizer; deterministic generator, no NPU."""
import asyncio,hashlib,json,os,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent
runtime=Path(os.environ['FNX_TEST_RUNTIME']) if os.environ.get('FNX_TEST_RUNTIME') else next(root for root in (P/'runtime',P,P.parent) if (root/'fnx').is_dir())
os.environ.update(FNX_PIPELINE='0',FNX_CTX='262144',FNX_ENABLE_THINKING='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
sys.path.insert(0,str(runtime))
model=types.ModuleType('fnx.model')
class Forbidden:
    def __init__(self,*a,**kw):raise AssertionError('REAL_MODEL_CONSTRUCTION_FORBIDDEN')
model.FnModel=Forbidden;sys.modules['fnx.model']=model
import torch
from transformers import AutoTokenizer
import fnx.server as S
torch.set_num_threads(1)
def forbidden(*a,**kw):raise AssertionError('REAL_ENGINE_CONSTRUCTION_FORBIDDEN')
S.Engine.__init__=forbidden
real=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next',trust_remote_code=True)
class Tokenizer:
    def __getattr__(self,k):return getattr(real,k)
    def __call__(self,*a,**kw):return real(*a,**kw)
    def apply_chat_template(self,messages,**kw):
        self.tools=[t['function']['name'] for t in kw.get('tools',[])]
        return real.apply_chat_template(messages,**kw)
class Fake:
    def __init__(self):self.tok=Tokenizer();self.stop_ids={248044,248046};self.n_req=0;self.last_prof={};self.calls=[]
    def generate(self,ids,limit,temp,tpl):
        toks=real.encode(self.fixture,add_special_tokens=False)
        if self.eos:toks.append(248046)
        toks=toks[:limit]
        self.calls.append(dict(limit=limit,prompt_tokens=len(ids),completion_tokens=len(toks),prompt_suffix=real.decode(ids)[-240:],tools=list(self.tok.tools)))
        for t in toks:yield ('t',t)
        self.last_prof={'n_prefill':len(ids),'tpot_ms':0}
fake=Fake();S.STATE['e']=fake
def tool(name,props):return dict(type='function',function=dict(name=name,parameters=dict(type='object',properties=props,required=list(props))))
echo=tool('mcp__audit__echo',{'text':{'type':'string'}})
typed=tool('audit_types',dict(count={'type':'integer'},enabled={'type':'boolean'},tags={'type':'array'},options={'type':'object'},label={'type':'string'}))
def xml(name,**params):return '<tool_call>\n<function='+name+'>\n'+''.join('<parameter='+k+'>'+v+'</parameter>\n' for k,v in params.items())+'</function>\n</tool_call>'
typed_xml=xml('audit_types',count='3',enabled='true',tags='["甲","乙"]',options='{"mode":"fast"}',label='123')
echo_xml=xml('mcp__audit__echo',text='你好，世界😀')
records=[]
async def request(name,fixture,**kw):
    fake.fixture=fixture;fake.eos=kw.pop('_eos',True)
    body=dict(model=S.MODEL_ID,messages=[dict(role='user',content='协议回归。')],max_completion_tokens=1024)
    body.update(kw)
    payload=json.dumps(body,ensure_ascii=False).encode();before=len(fake.calls)
    async def receive():return dict(type='http.request',body=payload,more_body=False)
    req=S.Request(dict(type='http',method='POST',path='/v1/chat/completions',headers=[(b'content-type',b'application/json')],query_string=b''),receive)
    r=await S.chat(req);record=dict(name=name)
    if isinstance(r,dict):record.update(status=200,response=r)
    elif hasattr(r,'body_iterator'):
        pieces=[]
        async for piece in r.body_iterator:pieces.append(piece.decode() if isinstance(piece,bytes) else piece)
        text=''.join(pieces);events=[]
        for line in text.splitlines():
            if line.startswith('data: ') and line!='data: [DONE]':events.append(json.loads(line[6:]))
        record.update(status=r.status_code,events=events,done=text.rstrip().endswith('data: [DONE]'))
    else:record.update(status=r.status_code,response=json.loads(r.body))
    record['engine_call']=fake.calls[-1] if len(fake.calls)>before else None
    records.append(record);return record
def calls(r):
    if 'events' in r:return [v for e in r['events'] for c in e.get('choices',[]) for v in c.get('delta',{}).get('tool_calls',[])]
    return r.get('response',{}).get('choices',[{}])[0].get('message',{}).get('tool_calls',[])
def text(r):
    if 'events' in r:return ''.join(c.get('delta',{}).get('content','') for e in r['events'] for c in e.get('choices',[]))
    return r['response']['choices'][0]['message'].get('content')
def finish(r):
    if 'events' in r:return next((c['finish_reason'] for e in reversed(r['events']) for c in e.get('choices',[]) if c.get('finish_reason')),None)
    return r['response']['choices'][0]['finish_reason']
async def main():
    for stream in (False,True):
        r=await request('typed_'+str(stream),typed_xml,tools=[typed],stream=stream)
        a=json.loads(calls(r)[0]['function']['arguments'])
        assert a==dict(count=3,enabled=True,tags=['甲','乙'],options={'mode':'fast'},label='123')
        assert type(a['count']) is int and type(a['enabled']) is bool and finish(r)=='tool_calls'
    r=await request('python_literals',typed_xml.replace('>true<','>True<').replace('{"mode":"fast"}',"{'mode':'fast'}"),tools=[typed])
    assert json.loads(calls(r)[0]['function']['arguments'])['enabled'] is True
    for stream in (False,True):
        r=await request('none_'+str(stream),echo_xml,tools=[echo],tool_choice='none',stream=stream)
        assert not calls(r) and text(r)==echo_xml and r['engine_call']['tools']==[] and finish(r)=='stop'
    for stream in (False,True):
        prefix='<tool_call>\n<function=mcp__audit__echo>\n'
        suffix=echo_xml[len(prefix):]
        r=await request('required_single_'+str(stream),suffix,tools=[echo],tool_choice='required',stream=stream)
        assert r['engine_call']['prompt_suffix'].endswith(prefix)
        assert calls(r)[0]['function']['name']=='mcp__audit__echo'
        assert json.loads(calls(r)[0]['function']['arguments'])==dict(text='你好，世界😀')
    r=await request('required_multi',echo_xml[len('<tool_call>\n'):],tools=[typed,echo],tool_choice='required')
    assert r['engine_call']['prompt_suffix'].endswith('<tool_call>\n') and calls(r)[0]['function']['name']=='mcp__audit__echo'
    r=await request('specific',echo_xml[len(prefix):],tools=[typed,echo],tool_choice=dict(type='function',function=dict(name='mcp__audit__echo')))
    assert r['engine_call']['tools']==['mcp__audit__echo'] and r['engine_call']['prompt_suffix'].endswith(prefix)
    assert calls(r)[0]['function']['name']=='mcp__audit__echo'
    r=await request('forced_model_restarted_envelope',echo_xml,tools=[echo],tool_choice='required',stream=True)
    assert len(calls(r))==1 and text(r)==''
    r=await request('specific_wrong_output_not_renamed',typed_xml,tools=[typed,echo],tool_choice=dict(type='function',function=dict(name='mcp__audit__echo')))
    assert not calls(r) and text(r)==typed_xml
    for name,kw in (
        ('unknown_function',dict(tools=[echo],tool_choice=dict(type='function',function=dict(name='missing')))),
        ('invalid_choice',dict(tools=[echo],tool_choice='bad')),
        ('required_no_tools',dict(tool_choice='required')),
        ('invalid_tools',dict(tools={})),
        ('invalid_stream_options',dict(stream_options=[])),
        ('invalid_usage_option',dict(stream_options={'include_usage':1})),
        ('invalid_modern_limit',dict(max_completion_tokens='bad'))):
        r=await request(name,'should not generate',**kw);assert r['status']==400 and r['engine_call'] is None
    r=await request('usage_and_indices',typed_xml+'\n'+echo_xml,tools=[typed,echo],stream=True,stream_options={'include_usage':True})
    assert r['done'] and [c['index'] for c in calls(r)]==[0,1]
    usage=[e for e in r['events'] if e.get('choices')==[]]
    assert len(usage)==1 and r['events'][-1] is usage[0]
    got=usage[0]['usage'];want=r['engine_call']
    assert got==dict(prompt_tokens=want['prompt_tokens'],completion_tokens=want['completion_tokens'],total_tokens=want['prompt_tokens']+want['completion_tokens'])
    assert all(e.get('usage') is None for e in r['events'][:-1])
    r=await request('default_usage_unchanged','plain',stream=True)
    assert not any('usage' in e for e in r['events']) and r['done']
    for stream in (False,True):
        r=await request('truncation_'+str(stream),typed_xml,tools=[typed],stream=stream,max_completion_tokens=32)
        assert finish(r)=='length' and not calls(r) and '<tool_call>' in text(r)
    r=await request('modern_only_large','测试。'*300);assert r['engine_call']['limit']==1024 and r['engine_call']['completion_tokens']>128
    r=await request('modern_precedence','ok',max_tokens=32,max_completion_tokens=512);assert r['engine_call']['limit']==512
    r=await request('null_legacy','ok',max_tokens=None,max_completion_tokens=512);assert r['engine_call']['limit']==512
    for stream in (False,True):
        value='你好，世界😀，普通聊天回归通过。'
        r=await request('unicode_'+str(stream),value,stream=stream);assert text(r)==value and finish(r)=='stop'
    r=await request('tool_result_roundtrip','收到。',tools=[echo],messages=[
        dict(role='user',content='请回声。'),dict(role='assistant',content=None,tool_calls=[dict(id='call_test',type='function',function=dict(name='mcp__audit__echo',arguments='{"text":"你好"}'))]),
        dict(role='tool',tool_call_id='call_test',content='你好'),dict(role='user',content='继续。')])
    assert r['status']==200 and text(r)=='收到。'
    assert not torch.npu.is_initialized()
    result=dict(status='passed',cases=len(records),npu_initialized=False,real_model_construction_forbidden=True,
        sources={n:hashlib.sha256((runtime/'fnx'/n).read_bytes()).hexdigest() for n in ('server.py','tool_protocol.py','request_validation.py')},records=records)
    output=os.environ.get('FNX_TEST_OUTPUT')
    if output or (P/'runtime/fnx').is_dir():
        folder=Path(output) if output else P;folder.mkdir(parents=True,exist_ok=True)
        (folder/'server-protocol-tests.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print('SERVER_TOOL_PROTOCOL_TESTS_PASSED',len(records),'NPU_INITIALIZED',False,flush=True)
asyncio.run(main())
