import json,tempfile
from pathlib import Path
from checkpoint import rows,append
with tempfile.TemporaryDirectory() as d:
 p=Path(d)/'checkpoint.jsonl';append(p,dict(case=1,text='通过'))
 good=p.read_bytes();p.write_bytes(good+b'{"case":2,"text":"\xe9')
 assert rows(p)==[dict(case=1,text='通过')] and p.read_bytes()==good
 assert p.with_suffix('.jsonl.interrupted-tail').exists()
 p.write_bytes(b'{"case":3}')
 assert rows(p)==[dict(case=3)] and p.read_bytes().endswith(b'\n')
 p.write_bytes(b'bad\n{"case":4}\n')
 try:rows(p)
 except json.JSONDecodeError:pass
 else:raise AssertionError('mid-file corruption must not be discarded')
print('DURABLE_CHECKPOINT_RECOVERY_PASS')
