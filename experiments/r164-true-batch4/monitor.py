import json,time
from pathlib import Path
P=Path(__file__).resolve().parent;w=json.loads((P/'window.json').read_text())
print('ELAPSED_S',round(time.time()-w['start_epoch'],1),'TO_RESTORE_S',round(w['restore_trigger_epoch']-time.time(),1))
for n in ('controller-nd.log','batch4-nd.log','batch1-nd.log','restore.log'):
    f=P/n
    if f.exists():
        lines=f.read_text(errors='replace').splitlines();print(n,'LINES',len(lines))
        print('\n'.join(s[:700] for s in lines[-8:] if len(s)<3000))
for b in (4,1):
    f=P/f'batch{b}-results.json'
    if f.exists():
        j=json.loads(f.read_text());print('RESULT',b,j['status'],[(r['requested_context'],r['rep'],round(r['total_decode_tps'],3) if r['total_decode_tps'] else None,[x['output_tokens'] for x in r['per_row']]) for r in j['records']])
    f=P/f'batch{b}-failure.json'
    if f.exists():print('FAILURE',b,f.read_text())
for n in ('restore-complete.json','final-validation.json'):
    if (P/n).exists():print(n,(P/n).read_text()[:200])
