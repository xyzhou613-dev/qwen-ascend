"""HTTP regression of the deployed model; no service or kernel changes."""
import ast
import concurrent.futures as cf
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import threading
import time
import urllib.request
import urllib.error
import uuid

SOURCE=Path(__file__).resolve().parent
RUN=Path(os.environ.get("FNX_REGRESSION_DIR","/tmp/fnx-http-regression"))
RUN.mkdir(parents=True,exist_ok=True)
BASE='http://127.0.0.1:8000'
RESULT={'version':'r120-service/r119-kernels','started_at':time.strftime('%F %T'),'cases':[]}


def save():
    (RUN/'results.json').write_text(json.dumps(RESULT,ensure_ascii=False,indent=2))


def record(group,name,ok,**details):
    row=dict(group=group,name=name,ok=bool(ok),**details)
    RESULT['cases'].append(row);save()
    print('CASE',group,name,'PASS' if ok else 'FAIL',flush=True)
    return row


def raw(path,body=None,timeout=120):
    data=body if isinstance(body,bytes) else (json.dumps(body,ensure_ascii=False).encode() if body is not None else None)
    req=urllib.request.Request(BASE+path,data=data,headers={'Content-Type':'application/json'})
    t=time.perf_counter()
    try:
        with urllib.request.urlopen(req,timeout=timeout) as r:
            b=r.read().decode('utf-8');status=r.status;ctype=r.headers.get('Content-Type','')
    except urllib.error.HTTPError as r:
        b=r.read().decode('utf-8',errors='replace');status=r.code;ctype=r.headers.get('Content-Type','')
    result=dict(status=status,content_type=ctype,seconds=time.perf_counter()-t,body=b)
    try:result['json']=json.loads(b)
    except ValueError:pass
    return result


def payload(text,max_tokens=64,stream=False,messages=None,**kw):
    return dict(model='qwen3.8-flash-next-w4',messages=messages or [{'role':'user','content':text}],
                max_tokens=max_tokens,temperature=0,stream=stream,enable_thinking=False,**kw)


def chat(body,timeout=120):
    r=raw('/v1/chat/completions',body,timeout)
    if r['status']!=200 or 'json' not in r:raise RuntimeError(str(r)[:1500])
    d=r['json'];d['client_seconds']=r['seconds'];return d


def answer(d):return d['choices'][0]['message']['content']


def stream(body,timeout=45):
    body=dict(body,stream=True)
    req=urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(body,ensure_ascii=False).encode(),headers={'Content-Type':'application/json'})
    text=[];events=[];stats=None;done=False;first=None;finish=None;t=time.perf_counter()
    with urllib.request.urlopen(req,timeout=timeout) as r:
        for line in r:
            if line.startswith(b': prefill '):
                print('PROGRESS',line.decode().strip(),round(time.perf_counter()-t,2),flush=True)
            if not line.startswith(b'data: '):continue
            data=line[6:].strip()
            if data==b'[DONE]':done=True;break
            d=json.loads(data);choice=d['choices'][0];piece=choice.get('delta',{}).get('content','')
            if piece:
                text.append(piece)
                if first is None:first=time.perf_counter()-t
            if choice.get('finish_reason') is not None:finish=choice['finish_reason']
            if 'fnx_stats' in d:stats=d['fnx_stats']
            events.append(d)
    return dict(text=''.join(text),ttft_s=first,seconds=time.perf_counter()-t,
                finish_reason=finish,done=done,stats=stats,events=events)


def monitor_health(stop,rows,timeout=.8):
    while not stop.is_set():
        t=time.perf_counter()
        try:
            r=raw('/health',timeout=timeout);rows.append(dict(ok=r['status']==200,seconds=r['seconds']))
        except Exception as exc:rows.append(dict(ok=False,seconds=time.perf_counter()-t,error=type(exc).__name__))
        stop.wait(.2)


def monitored(fn):
    stop=threading.Event();rows=[];thread=threading.Thread(target=monitor_health,args=(stop,rows),daemon=True)
    thread.start()
    try:value=fn()
    finally:stop.set();thread.join(3)
    return value,dict(samples=len(rows),failed=sum(not x['ok'] for x in rows),
                      max_seconds=max((x['seconds'] for x in rows),default=0),rows=rows)


def basic_mcq():
    spec=importlib.util.spec_from_file_location('mcq',SOURCE/'mcq_eval.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for i,(q,opts,key) in enumerate(m.QUESTIONS,1):
        text=q+'\n'+'\n'.join('ABCD'[j]+'. '+s for j,s in enumerate(opts))+'\n只回答正确选项的字母（单个字母），不要任何解释。'
        d=chat(payload(text,max_tokens=16));txt=answer(d)
        record('mcq',str(i),m.pick(txt)==key,question=q,expected=key,answer=txt,strict_format=txt.strip()==key,stats=d['fnx_stats'])


def code_cases():
    def code_timeout(*_):raise TimeoutError('generated function exceeded two seconds')
    signal.signal(signal.SIGALRM,code_timeout)
    cases=[
      ('gcd','gcd(a,b)，返回两个非负整数的最大公约数', [((48,18),6),((0,7),7),((17,13),1),((0,0),0)]),
      ('two_sum','two_sum(nums,target)，恰有一个解，返回两个不同元素的索引，按从小到大排列', [(([2,7,11,15],9),[0,1]),(([3,3],6),[0,1]),(([3,2,4],6),[1,2])]),
      ('stable_unique','stable_unique(items)，去除整数列表的重复项，保留第一次出现顺序', [(([3,1,3,2,1],),[3,1,2]),(([],),[]),(([1,1],),[1])]),
      ('is_prime','is_prime(n)，判断整数是否为质数，返回bool', [((2,),True),((1,),False),((9,),False),((97,),True),((-3,),False)]),
      ('binary_search','binary_search(nums,target)，在升序不重复整数列表中二分查找，返回下标，不存在返回-1', [(([1,3,5,7],5),2),(([],3),-1),(([1,3],2),-1),(([4],4),0)]),
      ('merge_intervals','merge_intervals(intervals)，合并相交或端点相接的闭区间，按起点升序返回二维列表', [(([[1,3],[2,6],[8,10]],),[[1,6],[8,10]]),(([],),[]),(([[4,5],[1,4]],),[[1,5]])]),
      ('is_palindrome','is_palindrome(s)，忽略非ASCII英文字母与数字，并忽略大小写，判断回文，返回bool', [(('A man, a plan, a canal: Panama',),True),(('race a car',),False),(('',),True)]),
      ('longest_common_prefix','longest_common_prefix(strs)，返回字符串列表的最长公共前缀，空列表返回空串', [((['flower','flow','flight'],),'fl'),(([],),''),((['dog','racecar','car'],),''),((['abc'],),'abc')]),
    ]
    builtins={'range':range,'len':len,'int':int,'str':str,'list':list,'dict':dict,'set':set,'tuple':tuple,
              'enumerate':enumerate,'zip':zip,'min':min,'max':max,'sum':sum,'abs':abs,'sorted':sorted,
              'all':all,'any':any,'bool':bool,'reversed':reversed}
    methods={'append','extend','sort','reverse','get','items','keys','values','fromkeys','lower','isalnum','isascii','add','join','count','startswith','pop'}
    for name,description,tests in cases:
        d=chat(payload('请写Python函数 '+description+'。仅输出函数代码，不要解释、不要import、不要类型注解。',max_tokens=384))
        txt=answer(d);match=re.search(r'```(?:python)?\s*\n(.*?)```',txt,re.S);code=match.group(1) if match else txt.strip()
        try:
            tree=ast.parse(code)
            fnames={n.name for n in tree.body if isinstance(n,ast.FunctionDef)}
            assert name in fnames
            assert all(isinstance(n,ast.FunctionDef) for n in tree.body),'top-level code'
            for n in ast.walk(tree):
                assert not isinstance(n,(ast.Import,ast.ImportFrom,ast.ClassDef,ast.Global,ast.Nonlocal,ast.With,ast.AsyncFunctionDef,ast.Try)),type(n).__name__
                if isinstance(n,ast.Attribute):assert n.attr in methods,n.attr
                if isinstance(n,ast.Name):assert not n.id.startswith('__'),n.id
                if isinstance(n,ast.Call):
                    assert (isinstance(n.func,ast.Name) and n.func.id in (set(builtins)|fnames)) or (isinstance(n.func,ast.Attribute) and n.func.attr in methods),'unapproved call'
            ns={'__builtins__':builtins};exec(compile(tree,'<model-code>','exec'),ns)
            outcomes=[]
            for args,expected in tests:
                # Inputs are tiny and bounded; the restricted AST cannot access I/O/imports.
                signal.setitimer(signal.ITIMER_REAL,2)
                try:got=ns[name](*json.loads(json.dumps(args)))
                finally:signal.setitimer(signal.ITIMER_REAL,0)
                try:json.dumps(got)
                except TypeError:got=repr(got)
                outcomes.append(dict(args=args,expected=expected,actual=got,ok=got==expected))
            record('code',name,all(x['ok'] for x in outcomes),code=code,outcomes=outcomes)
        except Exception as exc:record('code',name,False,code=code,error=repr(exc))


def stream_cases():
    for name,text in [('emoji','𠮷🧪🚀 café Ελληνικά'),('chinese','昇腾推理，边界测试。'),('english','Alpha beta 0123456789'),('json','{"status":"成功","emoji":"🧪"}')]:
        body=payload('请原样输出下面这一行，不添加解释：\n'+text,max_tokens=80)
        plain=chat(body);s=stream(body)
        record('stream',name,s['text']==answer(plain),plain=answer(plain),stream=s['text'],replacement_count=s['text'].count('\ufffd'),done=s['done'])
    body=payload('从1一直数到1000，逐项输出，不要跳过或解释。',max_tokens=1)
    plain=chat(body);s=stream(body)
    record('protocol','stream_finish_reason',s['finish_reason']==plain['choices'][0]['finish_reason'],
           nonstream_reason=plain['choices'][0]['finish_reason'],stream_reason=s['finish_reason'],
           plain=answer(plain),stream=s['text'],completion_tokens=plain['usage']['completion_tokens'])


def invalid_cases():
    base=payload('仅回答OK。',max_tokens=4)
    cases=[('invalid_json',b'{'),('nonobject_body',[]),('missing_messages',{'max_tokens':1}),
           ('bad_max_tokens',dict(base,max_tokens='bad')),('zero_max_tokens',dict(base,max_tokens=0)),
           ('negative_max_tokens',dict(base,max_tokens=-1)),('bad_temperature',dict(base,temperature='bad')),
           ('wrong_model',dict(base,model='qwen3.8-27b')),
           ('over_context_budget',dict(base,max_tokens=262145)),
           ('unsupported_n',dict(base,n=2))]
    for name,body in cases:
        r=raw('/v1/chat/completions',body)
        record('invalid_request',name,400<=r['status']<500,response=r)


def state_and_concurrency():
    a=payload('本次独立任务的口令是481726。只输出这个六位口令，不要解释。',max_tokens=16)
    b=payload('本次独立任务的口令是903158。只输出这个六位口令，不要解释。',max_tokens=16)
    r1=chat(a);r2=chat(a);rb=chat(b);r3=chat(a)
    record('state','repeat_and_switch_back',all('481726' in answer(x) and '903158' not in answer(x) for x in (r1,r2,r3)) and '903158' in answer(rb),
           answers=[answer(x) for x in (r1,r2,rb,r3)],repeat_prefill=r2['fnx_stats']['n_prefill'],switchback_prefill=r3['fnx_stats']['n_prefill'])
    record('cache','immediate_repeat',r2['fnx_stats']['n_prefill']==0,stats=r2['fnx_stats'])
    def concurrent():
        with cf.ThreadPoolExecutor(max_workers=3) as ex:
            fs=[]
            for code in ('571924','826315','439687'):
                p=payload('当前独立任务标识为'+code+'。第一行仅输出这个标识，随后从1逐项数到30，不要省略。',max_tokens=64)
                fs.append((code,ex.submit(stream,p)))
            return [(code,f.result()) for code,f in fs]
    values,h=monitored(concurrent)
    for code,r in values:
        record('concurrency',code,code in r['text'] and all(x not in r['text'] for x,_ in values if x!=code),
               text=r['text'],stats=r['stats'],done=r['done'])
    record('health','during_parallel_generation',h['failed']==0,**h)
    body=payload('请从1逐项写到1000，不能省略。',max_tokens=256,stream=True)
    req=urllib.request.Request(BASE+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    t=time.perf_counter();pieces=0
    with urllib.request.urlopen(req,timeout=120) as response:
        for line in response:
            if line.startswith(b'data: ') and line[6:].strip()!=b'[DONE]':
                d=json.loads(line[6:]);pieces+=bool(d['choices'][0].get('delta',{}).get('content'))
                if pieces>=3:break
    r=chat(payload('仅输出数字6174。',max_tokens=16),timeout=90)
    record('cancellation','disconnect_then_new_request','6174' in answer(r) and r['client_seconds']<5,
           aborted_after_pieces=pieces,follow_seconds=r['client_seconds'],follow_answer=answer(r),total_seconds=time.perf_counter()-t)
    rows=[]
    for i in range(30):
        code=str(700000+i);r=chat(payload('这是全新会话'+code+'。只输出 '+code+'，不做解释。',max_tokens=16))
        rows.append(dict(expected=code,answer=answer(r),seconds=r['client_seconds'],ok=code in answer(r)))
    record('stability','30_new_sessions',all(r['ok'] for r in rows),records=rows)


def long_cases():
    from transformers import AutoTokenizer
    tk=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next',local_files_only=True)
    corpus=Path('/root/ascend-dev/eval_text.txt').read_text();unit=tk(corpus+'\n').input_ids
    for ctx in (8192,262144):
        target=ctx-128;filler=(unit*(ctx//len(unit)+2))[:ctx];nonce=uuid.uuid4().hex;n=target-128
        facts=('项目海棠的验收码是7319。','项目松柏的验收码是8652。','项目银杏的验收码是2946。')
        for _ in range(15):
            parts=['本次文档编号：'+nonce+'。\n'];begin=0
            for point,fact in zip((n//10,n//2,n*9//10),facts):parts.extend([tk.decode(filler[begin:point]),'\n'+fact+'\n']);begin=point
            parts.extend([tk.decode(filler[begin:n]),'\n资料结束。请从资料提取海棠、松柏、银杏三个项目的验收码，按此顺序回答项目名与验收码。列完三个验收码后，再详细介绍资料中涉及的推理服务和长文档处理技术，连续写不少于800字。'])
            messages=[{'role':'user','content':''.join(parts)}]
            ids=tk(tk.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)).input_ids
            if len(ids)==target:break
            n+=target-len(ids)
        assert len(ids)==target
        body=payload('',max_tokens=128,messages=messages)
        r,h=monitored(lambda:stream(body,timeout=900))
        stats=r['stats'];hits=sum(code in r['text'] for code in ('7319','8652','2946'))
        record('long',str(ctx),hits==3 and stats['n_prefill']==target-4 and stats['tokens']==128 and r['done'],
               prompt_tokens=target,recall_hits=hits,text=r['text'],stats=stats,ttft_s=r['ttft_s'],seconds=r['seconds'])
        record('health','during_long_'+str(ctx),h['failed']==0,**h)
        repeated=chat(body,timeout=900)
        record('long_cache',str(ctx),repeated['fnx_stats']['n_prefill']==0 and all(code in answer(repeated) for code in ('7319','8652','2946')),
               text=answer(repeated),stats=repeated['fnx_stats'])
        follow=messages+[{'role':'assistant','content':'已收到资料。'},{'role':'user','content':'海棠的验收码是多少？仅输出四位数字。'}]
        d=chat(payload('',max_tokens=32,messages=follow),timeout=120)
        record('long_follow',str(ctx),'7319' in answer(d) and 0<d['fnx_stats']['n_prefill']<128,text=answer(d),stats=d['fnx_stats'])


def final_quality():
    r,h=monitored(lambda:raw('/fnx/loss',{'text':Path('/root/ascend-dev/eval_text.txt').read_text(),'max_tokens':200},timeout=180))
    d=r.get('json',{})
    record('quality','nll_200',r['status']==200 and math.isfinite(d.get('mean_nll',math.inf)) and abs(d.get('mean_nll',0)-2.164792)<.01,result=d,seconds=r['seconds'])
    record('health','during_loss_diagnostic',h['failed']==0,**h)
    final=chat(payload('仅回答数字2026。',max_tokens=16));record('health','final_generation','2026' in answer(final),answer=answer(final))


def main():
    RESULT['initial_health']=raw('/health');RESULT['initial_metrics']=raw('/metrics');save()
    for name,fn in [('mcq',basic_mcq),('code',code_cases),('stream',stream_cases),('invalid',invalid_cases),
                    ('state_concurrency_stability',state_and_concurrency),('long',long_cases),('quality',final_quality)]:
        print('PHASE',name,flush=True)
        try:fn()
        except Exception as exc:
            record('harness',name,False,error=repr(exc))
    RESULT['final_health']=raw('/health');RESULT['final_metrics']=raw('/metrics')
    RESULT['finished_at']=time.strftime('%F %T');save()
    print('REGRESSION_COMPLETE',len(RESULT['cases']),'failures',sum(not x['ok'] for x in RESULT['cases']),flush=True)


if __name__=='__main__':main()
