"""Prepare/install/restore only the five explicitly listed release files."""
import hashlib,json,os,shutil,stat,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;live=Path('/root/ascend-dev/fnx/fnx');startup=Path('/root/ascend-dev/p275.sh')
backup=P/'deployment-backup';stage=P/'release'
def sha(data):return hashlib.sha256(data).hexdigest()
def atomic(path,data,mode):
    tmp=path.with_name(path.name+'.r151-next');tmp.write_bytes(data);tmp.chmod(mode);os.replace(tmp,path)
def idle():
    code="from pathlib import Path; a=[]\nfor p in Path('/proc').iterdir():\n if not p.name.isdigit():continue\n try:\n  c=p.joinpath('cmdline').read_bytes().split(b'\\0')\n  if b'fnx.server' in c or b'fnx.pipeline_worker' in c:a.append(p.name)\n except (FileNotFoundError,PermissionError,ProcessLookupError):pass\nassert not a,('REFUSED_FILE_INSTALL_WITH_MODEL',a)"
    subprocess.run(['docker','exec','bole-model-Flash','/root/ascend-dev/venv/bin/python','-c',code],check=True)
mode=sys.argv[1]
if mode=='prepare':
    assert json.load(open(P/'ab-summary.json'))['accepted'],'AB gate not passed'
    assert not backup.exists(),'backup already exists'
    baseline=json.load(open(P/'production-before-hashes.json'))
    for name,digest in baseline.items():assert sha((live/name).read_bytes())==digest,name
    assert startup.read_bytes()==(P/'p275-before.sh').read_bytes()
    (stage/'fnx').mkdir(parents=True,exist_ok=True);backup.mkdir()
    for name in ('engine.py','prefill_output.py','libfnx_prefill_output.so'):shutil.copyfile(P/'candidate/fnx'/name,stage/'fnx'/name)
    raw=(live/'server.py').read_bytes();nl=b'\r\n' if b'\r\n' in raw else b'\n'
    a=b'"service_revision": "r150"';assert raw.count(a)==1;raw=raw.replace(a,b'"service_revision": "r151"')
    a=b'            "norm_post": os.environ.get("FNX_NORM_POST", "0") == "1",';assert raw.count(a)==1
    raw=raw.replace(a,a+nl+b'            "prefill_output": os.environ.get("FNX_PREFILL_OUTPUT", "0") == "1",')
    assert b'__r151_quality' not in raw and b'diagnostic_variant' not in raw
    (stage/'fnx/server.py').write_bytes(raw)
    raw=startup.read_bytes()
    a=b'FNX_NORM_POST=${FNX_NORM_POST:-1}\n';assert raw.count(a)==1
    raw=raw.replace(a,a+b'FNX_PREFILL_OUTPUT=${FNX_PREFILL_OUTPUT:-1}\n')
    a=b'FNX_NORM_POST=$FNX_NORM_POST ';assert raw.count(a)==1
    raw=raw.replace(a,a+b'FNX_PREFILL_OUTPUT=$FNX_PREFILL_OUTPUT ');(stage/'p275.sh').write_bytes(raw)
    entries=[]
    for name,target in [('engine.py',live/'engine.py'),('prefill_output.py',live/'prefill_output.py'),('libfnx_prefill_output.so',live/'libfnx_prefill_output.so'),('server.py',live/'server.py'),('p275.sh',startup)]:
        source=stage/('p275.sh' if name=='p275.sh' else 'fnx/'+name)
        old=target.read_bytes() if target.exists() else None
        if old is not None:(backup/name).write_bytes(old)
        entries.append(dict(name=name,target=str(target),stage=str(source),old_exists=old is not None,old_sha=sha(old) if old is not None else None,new_sha=sha(source.read_bytes()),mode=stat.S_IMODE(target.stat().st_mode) if old is not None else 0o755 if name.endswith('.so') else 0o644))
    (backup/'manifest.json').write_text(json.dumps(dict(state='prepared',entries=entries),indent=2));print('RELEASE_AND_EXACT_BACKUP_PREPARED')
elif mode in ('install','rollback'):
    idle();m=json.load(open(backup/'manifest.json'))
    for row in m['entries']:
        target=Path(row['target']);assert target==startup or target.parent==live
        current=sha(target.read_bytes()) if target.exists() else None
        assert current in (row['old_sha'],row['new_sha']),('external file change',str(target))
        if mode=='install':
            data=Path(row['stage']).read_bytes();assert sha(data)==row['new_sha'];atomic(target,data,row['mode'])
        elif row['old_exists']:
            data=(backup/row['name']).read_bytes();assert sha(data)==row['old_sha'];atomic(target,data,row['mode'])
        elif target.exists():target.unlink()
    m['state']='installed' if mode=='install' else 'rolled_back';(backup/'manifest.json').write_text(json.dumps(m,indent=2));print('RELEASE_FILES',m['state'])
else:raise SystemExit('unknown mode')
