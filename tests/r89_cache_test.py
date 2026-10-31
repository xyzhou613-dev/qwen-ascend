"""Run the actual cache policy without importing the NPU model or allocating tensors."""
import ast,gc,os,weakref
from pathlib import Path
from types import SimpleNamespace

package=Path(os.environ.get('FNX_TEST_PACKAGE',str(Path(__file__).resolve().parents[1]/'fnx')))
source=(package/'server.py').read_text(encoding='utf-8-sig')
cls=next(x for x in ast.parse(source).body if isinstance(x,ast.ClassDef) and x.name=='Engine')
method=next(x for x in cls.body if isinstance(x,ast.FunctionDef) and x.name=='_conv_make_room')
space={};exec(compile(ast.Module(body=[method],type_ignores=[]),'policy','exec'),space)
prepare=space['_conv_make_room']
class Snapshot:pass
released=[]
a,b,c=Snapshot(),Snapshot(),Snapshot()
obj=SimpleNamespace(conv_n=2,conv_slots=[a,b],conv_keys=[('a',),('b',)],conv_used=[9,2],
                    m=SimpleNamespace(release_state=lambda st:released.append(id(st))))
prepare(obj,a)
assert obj.conv_slots==[a,b] and not released
prepare(obj,c)
assert obj.conv_slots==[a] and obj.conv_keys==[('a',)] and obj.conv_used==[9]
assert released==[id(b)]
prepare(obj,c)
assert obj.conv_slots==[a] and released==[id(b)]
obj.conv_n=0;prepare(obj,c);assert obj.conv_slots==[a]
obj.conv_n=1;ref=weakref.ref(a);del a
prepare(obj,obj.conv_slots[0],preserve_selected=False);gc.collect()
assert ref() is None and not obj.conv_slots and not obj.conv_keys and not obj.conv_used
print('CACHE_EARLY_EVICTION_ACCEPT',flush=True)
