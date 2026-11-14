import json,subprocess,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
for attempt in range(20):
 rows=[]
 for _ in range(3):
  req=urllib.request.Request('http://127.0.0.1:9002/health',headers={'Connection':'close'})
  with urllib.request.urlopen(req,timeout=5) as r:h=json.load(r)
  assert h['service_revision']=='r152' and h['worker_alive']
  rows.append(h['pipeline_transfers']);time.sleep(1)
 sockets=subprocess.run(['ss','-tnH','sport = :9002'],capture_output=True,text=True,check=True).stdout.strip()
 if rows.count(rows[0])==3 and not sockets:
  with (P/'idle-check.jsonl').open('a') as f:f.write(json.dumps(dict(stable=True,attempt=attempt,transfers=rows[-1],sockets=sockets))+'\n')
  print('SERVICE_IDLE_VERIFIED',flush=True);break
else:raise SystemExit('BUSY_SERVICE: did not stop active requests')
