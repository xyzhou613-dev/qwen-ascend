import json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;d=json.loads((P/'original-private.json').read_text())
env=d['env'].copy();env['PYTHONPATH']=str(P)+os.pathsep+env.get('PYTHONPATH','');env['FNX_PIPELINE_LOG']=str(P/'profile-worker.log')
env['R145_CONTEXTS']=os.environ.get('R145_CONTEXTS','8k,256k')
raise SystemExit(subprocess.call([d['cmd'][0],str(P/sys.argv[1]),*sys.argv[2:]],cwd=P,env=env))
