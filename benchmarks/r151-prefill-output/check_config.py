from pathlib import Path
import json,re,sys
p=Path(__file__).resolve().parent;phase=sys.argv[1];assert phase in ('baseline','candidate');original=json.load(open(p/'original-private.json'))['env'];records=[]
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:
  cmd=proc.joinpath('cmdline').read_bytes().split(b'\0')
  role='server' if b'fnx.server' in cmd else 'worker' if b'fnx.pipeline_worker' in cmd else None
  if role is None:continue
  env=dict(x.decode().split('=',1) for x in proc.joinpath('environ').read_bytes().split(b'\0') if b'=' in x)
  assert env['FNX_PREFILL_OUTPUT']==('1' if phase=='candidate' else '0')
  keys=[k for k in original if k.startswith('FNX_') and k not in ('FNX_PORT','FNX_PIPELINE_LOG','FNX_PREFILL_OUTPUT') and not re.search(r'API|SECRET|PASSWORD|CREDENTIAL|AUTH|_KEY',k)]
  assert all(env.get(k)==original[k] for k in keys),[k for k in keys if env.get(k)!=original[k]]
  records.append(dict(role=role,pid=int(proc.name),matching_inference_settings=len(keys),prefill_output=phase=='candidate',visible=env['ASCEND_RT_VISIBLE_DEVICES']))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
assert len(records)==2 and {r['visible'] for r in records}=={'0','1'}
(p/f'{phase}-config-check.json').write_text(json.dumps(records,indent=2));print('BOTH_PROCESS_SETTINGS_CONFIRMED',records,flush=True)
