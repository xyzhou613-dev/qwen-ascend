"""Replay captured model arguments through the candidate without inference."""
import hashlib,json,os,sys,types
from pathlib import Path
P=Path(__file__).resolve().parent;runtime=Path(os.environ.get('FNX_TEST_RUNTIME',P/'runtime'))
pkg=types.ModuleType('fnx');pkg.__path__=[str(runtime/'fnx')];sys.modules['fnx']=pkg
from fnx.tool_protocol import parse_tool_calls
audit=Path('/models/fnx-diagnostics/tool-protocol-audit-20261114/live-audit.json')
records=[]
for r in json.loads(audit.read_text())['records']:
    if r['name'] not in ('typed_nonstream','unicode_sse_usage','long_modern_limit'):continue
    texts=[];before=[]
    for call in r['summary']['calls']:
        args=json.loads(call['function']['arguments']);before.append({k:type(v).__name__ for k,v in args.items()})
        texts.append('<tool_call>\n<function='+call['function']['name']+'>\n'+''.join('<parameter='+k+'>'+v+'</parameter>\n' for k,v in args.items())+'</function>\n</tool_call>')
    clean,calls=parse_tool_calls('\n'.join(texts),r['request']['tools']);assert calls and not clean
    args=[json.loads(c['function']['arguments']) for c in calls]
    if r['name']=='typed_nonstream':assert {k:type(v).__name__ for k,v in args[0].items()}==dict(count='int',enabled='bool',tags='list',options='dict',label='str')
    if r['name']=='long_modern_limit':assert isinstance(args[0]['todos'],list) and len(args[0]['todos'])==6
    if r['name']=='unicode_sse_usage':assert args[0]['text']=='你好，世界😀'
    records.append(dict(case=r['name'],before_types=before,after_types=[{k:type(v).__name__ for k,v in a.items()} for a in args]))
assert len(records)==3
(P/'live-output-replay.json').write_text(json.dumps(dict(status='passed',records=records,scope='captured arguments, no inference',
    source_sha256=hashlib.sha256((runtime/'fnx/tool_protocol.py').read_bytes()).hexdigest()),ensure_ascii=False,indent=2))
print('CAPTURED_MODEL_ARGUMENT_REPLAY_PASS',len(records))
