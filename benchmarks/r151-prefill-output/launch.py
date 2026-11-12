import json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;d=json.loads((P/'original-private.json').read_text())
env=d['env'].copy()
if sys.argv[1]=='--server':
    variant=sys.argv[2];assert variant in ('baseline','candidate')
    active=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:
            cmd=p.joinpath('cmdline').read_bytes().split(b'\0')
            if b'fnx.server' in cmd or b'fnx.pipeline_worker' in cmd:active.append(p.name)
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    assert not active,('REFUSED_SECOND_ENGINE',active)
    cwd=P/variant;env['PYTHONPATH']=str(cwd)+os.pathsep+str(P)+os.pathsep+env.get('PYTHONPATH','')
    env['FNX_PREFILL_OUTPUT']='1' if variant=='candidate' else '0'
    env['R151_VARIANT']=variant;env['FNX_PORT']='19002';env['FNX_PIPELINE_LOG']=str(P/f'{variant}-worker.log')
    with (P/f'{variant}-server.log').open('ab') as f:
        child=subprocess.Popen(d['cmd'],cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
    (P/f'{variant}.pid').write_text(str(child.pid));print('DIAGNOSTIC_STARTED',variant,child.pid,flush=True)
else:
    assert sys.argv[1]=='layer_probe.py'
    env['PYTHONPATH']=str(P/'candidate')+os.pathsep+str(P)+os.pathsep+env.get('PYTHONPATH','')
    env['ASCEND_RT_VISIBLE_DEVICES']='0,1';env['FNX_PREFILL_OUTPUT']='1'
    raise SystemExit(subprocess.call([d['cmd'][0],str(P/sys.argv[1])],cwd=P,env=env))
