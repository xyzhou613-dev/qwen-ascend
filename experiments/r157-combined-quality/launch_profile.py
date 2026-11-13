import json,os,subprocess,sys
from pathlib import Path
P=Path(__file__).resolve().parent;phase=sys.argv[1];assert phase in ('combined','smooth');cwd=P/('profile-'+phase)
d=json.load(open(P/'original-private.json'));env=d['env'].copy()
env.update(PYTHONPATH=str(cwd),FNX_DENSE_W8A8='1',FNX_KMAJOR_DOWN='1' if phase=='combined' else '0',
 FNX_KMAJOR_SO=str(cwd/'fnx/libfnx_w4_kmajor.so'),FNX_W8A8_CALIB_DIR='/models/fnx-diagnostics/r154-w8a8-gemv-20261113/private-calibration',
 FNX_PIPELINE_LOG=str(cwd/'worker.log'),R145_CONTEXTS='8k,256k')
with (cwd/'profile.log').open('ab') as f:
 child=subprocess.Popen([sys.executable,str(cwd/'profile_model.py')],cwd=cwd,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
(P/f'profile-{phase}.pid').write_text(str(child.pid));print('PROFILE_STARTED',phase,child.pid,flush=True)
