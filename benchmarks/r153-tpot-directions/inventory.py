from pathlib import Path
import json,struct,math,hashlib
P=Path(__file__).resolve().parent;src=Path('/models/Qwen3.8-Flash-Next');idx=json.load(open(src/'model.safetensors.index.json'))['weight_map'];headers={};rows=[]
for key,file in idx.items():
 if not key.startswith('mtp.'):continue
 if file not in headers:
  with (src/file).open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];headers[file]=json.loads(f.read(n))
 d=headers[file][key];num=math.prod(d['shape']);rows.append(dict(key=key,shape=d['shape'],dtype=d['dtype'],parameters=num,bytes=d['data_offsets'][1]-d['data_offsets'][0]))
cpp=Path('/root/ascend-dev/venv/lib/python3.12/site-packages/transformers/models/qwen4_exp/modeling_qwen4_exp.py').read_text()
r=dict(mtp=rows,total_parameters=sum(x['parameters'] for x in rows),total_bf16_bytes=sum(x['bytes'] for x in rows),expert_parameters=sum(x['parameters'] for x in rows if '.experts.' in x['key']),transformers_ignores_mtp='_keys_to_ignore_on_load_unexpected = [r"^mtp.*"' in cpp,production_revision='r152')
(P/'mtp-inventory.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
live=Path('/root/ascend-dev/fnx/fnx');(P/'production-before-hashes.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in live.iterdir() if f.is_file() and f.suffix in ('.py','.so')},indent=2))
