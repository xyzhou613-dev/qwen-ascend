import json,subprocess,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent
for unit in ('fnx-r159-run.service','fnx-r159-safety.service','fnx-r159-rescue.service','fnx-r159-rescue.timer'):
    print(unit,subprocess.run(['systemctl','show',unit,'-p','ActiveState','-p','SubState','-p','Result'],capture_output=True,text=True).stdout.strip().replace('\n',' '))
if (P/'window-start.txt').exists():print('ELAPSED_SECONDS',round(time.time()-float((P/'window-start.txt').read_text()),1))
for name in ('component-device0.json','component-device1.json','component-selection.json','layers-device0.json','layers-device1.json','baseline/summary.json','combined/summary.json','restore-complete.json'):
    f=P/name
    if not f.exists():continue
    d=json.loads(f.read_text());print(name,{k:v for k,v in d.items() if k in ('status','median_us','selected_tile','passed','median_8k_tpot_ms','median_256k_tpot_ms','elapsed_s','within_hour')})
for phase in ('baseline','combined'):
    f=P/f'{phase}-prefill-quality.json'
    if f.exists():print(phase,'LEGACY_PREFILL',{k:round(v['mean_nll'],8) for k,v in json.loads(f.read_text()).items()})
for name in ('controller.log','component-device0.log','component-device1.log','layers-device0.log','layers-device1.log'):
    f=P/name
    if f.exists():
        tail=f.read_text(errors='replace').splitlines()[-3:]
        print(name,'\n'.join(tail))
