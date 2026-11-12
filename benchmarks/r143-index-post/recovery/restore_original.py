import json,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent;d=json.loads((p/'original-private.json').read_text())
r=subprocess.run(['pgrep','-f','^/root/ascend-dev/venv/bin/python -m fnx.server$'],capture_output=True,text=True)
if r.returncode==0:
 print('RESTORE_ALREADY_RUNNING',r.stdout.strip());raise SystemExit(0)
with open('/root/ascend-dev/logs/server.log','ab') as f:
 child=subprocess.Popen(d['cmd'],cwd=d['cwd'],env=d['env'],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (p/'restored.pid').write_text(str(child.pid));print('RESTORED_ORIGINAL_PID',child.pid)
