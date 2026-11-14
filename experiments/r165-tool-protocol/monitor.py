import json,time
from pathlib import Path
P=Path(__file__).resolve().parent;w=json.loads((P/'window.json').read_text())
print('ELAPSED',round(time.time()-w['start_epoch'],1),'TO_ROLLBACK',round(w['rollback_trigger_epoch']-time.time(),1))
for n in ('deployment.log','candidate-start.log','live-tests.log','rollback-start.log'):
    f=P/n
    if f.exists():print(n,'\n'+'\n'.join(s[:600] for s in f.read_text(errors='replace').splitlines()[-8:] if len(s)<3000))
for n in ('deployment-error.json','accepted.json','rollback-complete.json','boot-rollback-complete.json'):
    if (P/n).exists():print(n,(P/n).read_text()[:900])
