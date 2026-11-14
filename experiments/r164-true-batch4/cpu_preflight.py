"""Exercise the actual prefix-view context on CPU and real tokenizer inputs."""
import ast,contextlib,json,os,types
from pathlib import Path
import torch
from transformers import AutoTokenizer
P=Path(__file__).resolve().parent
tree=ast.parse((P/'bench.py').read_text())
nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in ('prefix_row','prompts')]
E=types.SimpleNamespace(B=4)
env=dict(contextlib=contextlib,os=os,E=E,sync=lambda:None,B=4)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(P/'bench.py'),'exec'),env)
def fixture(b):
    stores=[types.SimpleNamespace(k=[torch.zeros(b,2,8,3)],v=[torch.zeros(b,2,8,3)]) for _ in range(2)]
    layers=[]
    for st in stores:
        for full in (False,True):
            l=types.SimpleNamespace(conv=torch.zeros(b,5,3),index_k=None,index_keys=None)
            if full:l.index_k=torch.zeros(b,8,3);l.index_keys=torch.zeros(b,2,3)
            else:l.S=torch.zeros(b,2,3,3)
            layers.append(l)
    m=types.SimpleNamespace(kv0=stores[0],kv1=stores[1],layers0=layers[:2],layers1=layers[2:])
    return m,stores,layers
for b in (1,4):
    E.B=b;m,stores,layers=fixture(b)
    original=[(l,{k:v for k,v in vars(l).items() if torch.is_tensor(v)}) for l in layers]
    for failure in (False,True):
        try:
            with env['prefix_row'](m):
                assert E.B==1
                assert m.kv0.k[0].shape[0]==1
                for l in layers:
                    for k,v in vars(l).items():
                        if torch.is_tensor(v):v.fill_(7)
                for st in stores:
                    for t in st.k+st.v:t.fill_(9)
                if failure:raise ValueError('TEST_RESTORE_AFTER_FAILURE')
        except ValueError:assert failure
        assert E.B==b
        for l,attrs in original:
            for k,v in attrs.items():assert getattr(l,k) is v and torch.equal(v,torch.full_like(v,7))
        for st in stores:
            for t in st.k+st.v:assert t.shape[0]==b and torch.equal(t,torch.full_like(t,9))
tok=AutoTokenizer.from_pretrained('/models/Qwen3.8-Flash-Next',trust_remote_code=True)
checks=[]
for b in (1,4):
    env['B']=b
    for n in (8000,261900):
        rows,common=env['prompts'](tok,n)
        assert common>=n-8 and common<len(rows[0])
        checks.append(dict(batch=b,target=n,input_lengths=list(map(len,rows)),common_prefix=common))
(P/'cpu-preflight.json').write_text(json.dumps(dict(status='passed',prefix_views_success_exception_b1_b4=True,prompts=checks),indent=2))
print(json.dumps(checks))
