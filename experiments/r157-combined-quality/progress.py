"""Read-only progress: never repairs a checkpoint while its writer is active."""
import json,pathlib,subprocess,time
P=pathlib.Path(__file__).resolve().parent
def rows(path):
 if not path.exists():return []
 out=[]
 for line in path.read_bytes().splitlines(keepends=True):
  if not line.endswith(b'\n'):continue
  try:out.append(json.loads(line))
  except (json.JSONDecodeError,UnicodeDecodeError):pass
 return out
def data(path):
 try:return json.loads(path.read_text())
 except (FileNotFoundError,json.JSONDecodeError):return []
starts=[P/f'{p}-start.txt' for p in ('baseline','smooth','combined','rollback') if (P/f'{p}-start.txt').exists()];phase=max(starts,key=lambda x:x.stat().st_mtime).name[:-10] if starts else 'preparing'
unit=(P/'current-run-unit').read_text().strip() if (P/'current-run-unit').exists() else 'fnx-r157-run'
state=subprocess.check_output(['systemctl','show',unit,'--value','-p','ActiveState'],text=True).strip()
bench=[]
for name,label in [('8k.json','8k'),('256k.json','long')]:
 r=data(P/phase/name);bench.append(f'{label}={len(r)}/3')
ns={p:len(rows(P/f'{p}-nll.jsonl')) for p in ('baseline','smooth','combined','rollback')}
ts={p:len(rows(P/f'quality-{p}.jsonl')) for p in ('combined','rollback')}
boot_changed=(P/'original-boot-id').exists() and (P/'original-boot-id').read_text().strip()!=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip()
reference=rows(P/'canonical-grader.jsonl')
remaining=(int((P/'window-start.txt').read_text())+20100-time.time())/60 if (P/'window-start.txt').exists() else None
remaining_text=f'{remaining:.1f}' if remaining is not None else 'not_started'
print(f'{phase} | {state} | '+','.join(bench)+f' | NLL={ns} | tasks={ts} | refs={len(reference)}/1200,failed={sum(not r["result"]["correct"] for r in reference)} | restore_in_min={remaining_text} | boot_changed={boot_changed}')
log=P/'controller.log'
if log.exists():
 lines=log.read_text(errors='replace').splitlines()
 print((lines[-1] if lines else '')[:220])
