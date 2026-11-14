"""Freeze tested protocol files and backup the still-online canonical source."""
import hashlib,json,shutil,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent;src=Path('/root/ascend-dev/fnx/fnx');sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
j=json.loads((P/'server-protocol-tests.json').read_text());assert j['status']=='passed' and j['cases']==28 and not j['npu_initialized']
assert all(sha(P/'runtime/fnx'/n)==h for n,h in j['sources'].items())
assert 'Ran 39 tests' in (P/'unit-tests.log').read_text()
r=json.loads((P/'live-output-replay.json').read_text());assert r['status']=='passed' and r['source_sha256']==sha(P/'runtime/fnx/tool_protocol.py')
assert not (src/'tool_protocol.py').exists()
assert sha(src/'server.py')=='041819b4bd1c0345739a7646cb7e8e2fd50a2087c1c161a4d2b126137cbb1b01'
assert sha(src/'request_validation.py')=='9e69d7ba4224c74c3e8214cee5648dd139ffa0c172cfff7de2479f0fea845ebc'
old={str(f.relative_to(src)):sha(f) for f in src.rglob('*') if f.is_file() and '__pycache__' not in f.parts and f.suffix in ('.py','.so','.cpp','.h','.c')}
assert len(old)==45
backup=P/'backup';backup.mkdir(exist_ok=True)
for n in ('server.py','request_validation.py'):
    f=backup/n;assert not f.exists() or sha(f)==old[n];shutil.copy2(src/n,f)
m=dict(epoch=time.time(),old_files=old,new_files=j['sources'],p275_sha256=sha(Path('/root/ascend-dev/p275.sh')),
    base_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/root/qwen-ascend',text=True).strip(),unit_tests=39,handler_tests=28,
    user_request='请帮我修复他',maintenance_basis='Current repair instruction and original task authorization for necessary short service maintenance',
    container_id=subprocess.check_output(['docker','inspect','-f','{{.Id}}','bole-model-Flash'],text=True).strip())
(P/'release-manifest.json').write_text(json.dumps(m,ensure_ascii=False,indent=2))
print('RELEASE_BACKUP_AND_TESTED_HASHES_READY',len(old),len(m['new_files']))
