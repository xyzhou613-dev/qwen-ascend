"""Durable JSONL checkpoints, repairing only an interrupted final record."""
import json,os
def rows(path):
 if not path.exists():return []
 raw=path.read_bytes();parts=raw.splitlines(keepends=True);result=[];prefix=b''
 for i,line in enumerate(parts):
  try:value=json.loads(line)
  except (json.JSONDecodeError,UnicodeDecodeError):
   if i!=len(parts)-1:raise
   path.with_suffix(path.suffix+'.interrupted-tail').write_bytes(line);path.write_bytes(prefix);break
  result.append(value);prefix+=line if line.endswith(b'\n') else line+b'\n'
 else:
  if prefix!=raw:path.write_bytes(prefix)
 return result
def append(path,value):
 with path.open('a',encoding='utf-8') as f:
  f.write(json.dumps(value,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
