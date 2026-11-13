import json,os,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent;env=os.environ.copy();env.update(json.load(open(p/'production-selected-config.json')));env['ASCEND_RT_VISIBLE_DEVICES']=sys.argv[1];env['PYTHONPATH']='/root/ascend-dev/fnx'+os.pathsep+str(p)+os.pathsep+str(p/'python_deps')+os.pathsep+env.get('PYTHONPATH','');env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
raise SystemExit(subprocess.call([sys.executable,str(p/sys.argv[2])]+sys.argv[3:],cwd=p,env=env))
