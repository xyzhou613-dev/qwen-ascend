import json,os,subprocess,sys
from pathlib import Path
p=Path(__file__).resolve().parent;assert sys.argv[1] in ('0','1')
env=os.environ.copy();env.update(json.load(open(p/'production-selected-config.json')))
env['ASCEND_RT_VISIBLE_DEVICES']=sys.argv[1];env['PYTHONPATH']=str(p/'candidate')+os.pathsep+str(p)+os.pathsep+env.get('PYTHONPATH','')
env['FNX_NORM_ORDERED']='0';env['OMP_NUM_THREADS']='1';env['MKL_NUM_THREADS']='1'
raise SystemExit(subprocess.call([sys.executable,str(p/'all_linear_probe.py'),sys.argv[1]],cwd=p,env=env))
