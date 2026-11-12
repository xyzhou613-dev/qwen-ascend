"""Checked, reversible installation of the selected runtime files."""
import hashlib,json,shutil,sys
from pathlib import Path
P=Path(__file__).resolve().parent;root=Path('/root/ascend-dev');B=P/'deployment-backup'
files=['fnx/fnx/engine.py','fnx/fnx/server.py','fnx/fnx/norm_post.py','fnx/fnx/libfnx_norm_post.so','p275.sh']
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
if sys.argv[1]=='install':
 assert not B.exists(),'backup already exists'
 B.mkdir();manifest={}
 for f in files:
  src=root/f;bak=B/f;bak.parent.mkdir(parents=True,exist_ok=True)
  if src.exists():shutil.copy2(src,bak)
  manifest[f]={'before':sha(src)}
 # Reject concurrent modifications to the existing tracked source files.
 for f in ('engine.py','server.py'):
  assert (root/'fnx/fnx'/f).read_bytes()==(Path('/root/qwen-ascend/fnx')/f).read_bytes(),f
 assert '/__r144_quality' not in (P/'release/fnx/server.py').read_text()
 startup=(root/'p275.sh').read_text()
 anchor='FNX_INDEX_POST=${FNX_INDEX_POST:-1}'
 assert startup.count(anchor)==1
 startup=startup.replace(anchor,anchor+'\nFNX_NORM_POST=${FNX_NORM_POST:-1}\nFNX_DELTA_REUSE=${FNX_DELTA_REUSE:-1}')
 anchor='FNX_CONV_PREFILL=$FNX_CONV_PREFILL FNX_INDEX_POST=$FNX_INDEX_POST'
 assert startup.count(anchor)==1
 startup=startup.replace(anchor,'FNX_NORM_POST=$FNX_NORM_POST FNX_DELTA_REUSE=$FNX_DELTA_REUSE '+anchor)
 for name in ('engine.py','server.py','norm_post.py','libfnx_norm_post.so'):manifest['fnx/fnx/'+name]['installed']=sha(P/'release/fnx'/name)
 manifest['p275.sh']['installed']=hashlib.sha256(startup.encode()).hexdigest()
 # Persist the rollback manifest before the first mutation.
 (B/'manifest.json').write_text(json.dumps(manifest,indent=2))
 for name in ('engine.py','server.py','norm_post.py','libfnx_norm_post.so'):shutil.copy2(P/'release/fnx'/name,root/'fnx/fnx'/name)
 (root/'p275.sh').write_text(startup)
 print('FILES_INSTALLED',flush=True)
elif sys.argv[1]=='rollback':
 manifest=json.loads((B/'manifest.json').read_text())
 for f,row in manifest.items():assert sha(root/f) in (row['before'],row['installed']),(f,'changed outside task')
 for f,row in manifest.items():
  if row['before'] is None:(root/f).unlink(missing_ok=True)
  else:shutil.copy2(B/f,root/f)
 print('FILES_ROLLED_BACK',flush=True)
