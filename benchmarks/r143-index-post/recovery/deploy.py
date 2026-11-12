from pathlib import Path
import hashlib,json,runpy,shutil,statistics,subprocess
P=Path(__file__).resolve().parent;R=Path('/root/qwen-ascend');runtime=Path('/root/ascend-dev/fnx/fnx')
log=(P/'controller.log').read_text();assert 'CANDIDATE_ACCEPT_COMPLETE' in log and 'R142_AND_MANAGER_RESTORED' in log
rows=json.loads((P/'probe.json').read_text());assert all(r['exact'] for r in rows if r.get('mode')==1 and r['kind']!='timing')
layers=json.loads((P/'full-layer.json').read_text());assert len(layers)==4 and all(all(r['exact'].values()) for r in layers)
quality=json.loads((P/'candidate-prefill-quality.json').read_text());assert quality['mean_nll']==1.722880 and quality['scored']==199
for i in (1,2,3):assert (P/f'baseline/answer{i}.txt').read_bytes()==(P/f'candidate8k/answer{i}.txt').read_bytes()
for n in ('engine.py','index_post.py'):assert (R/'fnx'/n).read_bytes()==(P/'fnx'/n).read_bytes()
assert (R/'fnx/server.py').read_bytes()==(P/'server-clean.py').read_bytes()
assert b'__r143_prefill_quality' not in (R/'fnx/server.py').read_bytes()
profile=Path('/root/ascend-dev/p275.sh');patch=runpy.run_path(str(P/'r143-enable.py'))['patch_profile']
patched=patch(profile.read_bytes());assert patch(patched)==patched;subprocess.run(['bash','-n'],input=patched,check=True)
B=P/'deployment-backup';assert not B.exists();B.mkdir(mode=0o700)
targets=[runtime/n for n in ('engine.py','server.py','index_post.py','libfnx_index_post.so')]+[profile]
manifest=[]
for i,f in enumerate(targets):
 row=dict(target=str(f),backup=str(B/str(i)),existed=f.exists())
 if f.exists():shutil.copy2(f,B/str(i));row['sha256']=hashlib.sha256(f.read_bytes()).hexdigest()
 manifest.append(row)
(B/'manifest.json').write_text(json.dumps(manifest,indent=2))
for n in ('engine.py','server.py','index_post.py'):shutil.copy2(R/'fnx'/n,runtime/n)
shutil.copy2(P/'fnx/libfnx_index_post.so',runtime/'libfnx_index_post.so');profile.write_bytes(patched)
(P/'deployment-manifest.json').write_text(json.dumps(dict(revision='r143',files={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in targets}),indent=2))
print('DEPLOYED_EXACT_FILES_AND_PROFILE',flush=True)
