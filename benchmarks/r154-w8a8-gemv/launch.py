import json,os,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent;env=os.environ.copy()
env.update(json.load(open(p/'production-selected-config.json')))
env['ASCEND_RT_VISIBLE_DEVICES']=sys.argv[1]
env['PYTHONPATH']=str(p/'candidate')+os.pathsep+str(p)+os.pathsep+env.get('PYTHONPATH','')
env['OMP_NUM_THREADS']=env['MKL_NUM_THREADS']='1'
env['FNX_W8A8_LAYOUT']='nd'
raise SystemExit(subprocess.call([sys.executable,str(p/sys.argv[2])]+sys.argv[3:],cwd=p,env=env))
