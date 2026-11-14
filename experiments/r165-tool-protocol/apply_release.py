"""Atomically install or restore only the three protocol files."""
import hashlib,json,os,sys,uuid,shutil,tempfile
from pathlib import Path
P=Path(__file__).resolve().parent;dest=Path('/root/ascend-dev/fnx/fnx')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
m=json.loads((P/'release-manifest.json').read_text())
def replace(path,data):
    tmp=path.with_name('.'+path.name+'.r165-'+uuid.uuid4().hex+'.tmp')
    with tmp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.chmod(tmp,0o644);os.replace(tmp,path)
def apply(mode,target):
    if mode=='install':
        assert all(sha(target/n)==h for n,h in m['old_files'].items())
        assert not (target/'tool_protocol.py').exists()
        for n in ('tool_protocol.py','request_validation.py','server.py'):
            src=P/'runtime/fnx'/n;assert sha(src)==m['new_files'][n];replace(target/n,src.read_bytes())
    elif mode=='rollback':
        for n in ('server.py','request_validation.py'):
            assert sha(target/n) in (m['new_files'][n],m['old_files'][n]),'UNEXPECTED_CONCURRENT_SOURCE_EDIT'
            src=P/'backup'/n;assert sha(src)==m['old_files'][n];replace(target/n,src.read_bytes())
        added=target/'tool_protocol.py'
        if added.exists():assert sha(added)==m['new_files']['tool_protocol.py'];added.unlink()
    else:raise ValueError(mode)
if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='selftest':
        with tempfile.TemporaryDirectory(prefix='rollback-sandbox-',dir=P) as folder:
            target=Path(folder).resolve();assert target.is_relative_to(P.resolve())
            for n in m['old_files']:
                f=target/n;f.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(dest/n,f)
            apply('install',target)
            assert all(sha(target/n)==v for n,v in m['new_files'].items())
            apply('rollback',target)
            assert all(sha(target/n)==v for n,v in m['old_files'].items()) and not (target/'tool_protocol.py').exists()
        assert all(sha(dest/n)==v for n,v in m['old_files'].items())
        (P/'rollback-dryrun.json').write_text(json.dumps(dict(status='passed',old_bytes_restored=True,production_unchanged=True,source_sha256=sha(Path(__file__))),indent=2))
    else:
        assert (P/'lease.active').exists();apply(mode,dest)
    print('PROTOCOL_SOURCE_'+mode.upper(),flush=True)
