import pathlib,shutil
P=pathlib.Path(__file__).resolve().parent;OLD=pathlib.Path('/models/fnx-diagnostics/r155-smooth-tpot-ab-20261113')
for name in ('snapshot.py','stop_owned.py','idle_guard.py','http_bench.py','formal_smoke.py'):
 shutil.copyfile(OLD/name,P/name)
f=P/'stop_owned.py';s=f.read_text();s=s.replace("captured=(P/'stop-requested').exists()", "captured=(P/'original-boot-id').read_text().strip()==Path('/proc/sys/kernel/random/boot_id').read_text().strip() and (P/'stop-requested').exists()")
s=s.replace("(str(P/'baseline'),str(P/'candidate'))", "tuple(str(P/n) for n in ('baseline','smooth','combined','profile-combined','profile-smooth'))")
s=s.replace("auxiliary=str(P/'layer_probe.py').encode() in cmd", "auxiliary=any(str(P/n/'profile_model.py').encode() in cmd for n in ('profile-combined','profile-smooth'))")
f.write_text(s)
link=P/'private-quality'
if not link.exists():link.symlink_to('/models/fnx-diagnostics/r156-moe-kmajor-20261113/private-quality',target_is_directory=True)
(P/'inputs').mkdir(exist_ok=True)
for f in (OLD/'inputs').glob('*.json'):shutil.copyfile(f,P/'inputs'/f.name)
R145=pathlib.Path('/root/qwen-ascend/benchmarks/r145-hardware-profile')
for phase in ('combined','smooth'):
 d=P/('profile-'+phase);d.mkdir(exist_ok=True)
 source=(pathlib.Path('/models/fnx-diagnostics/r156-moe-kmajor-20261113/candidate/fnx') if phase=='combined' else pathlib.Path('/models/fnx-diagnostics/r154-w8a8-gemv-20261113/selected/fnx'))
 shutil.copytree(source,d/'fnx')
 for name in ('kmajor_down.py','libfnx_w4_kmajor.so'):shutil.copyfile('/models/fnx-diagnostics/r156-moe-kmajor-20261113/'+name,d/'fnx'/name)
 for name in ('profile_support.py','profile_model.py','patch.py'):shutil.copyfile(R145/name,d/name)
 f=d/'profile_model.py';s=f.read_text();start=s.index("   ref=(Path(");end=s.index('\n   quality=',start)
 s=s[:start]+"   ref=None"+s[end:];s=s.replace('reference_starts_with_output=ref.startswith(answer)','reference_starts_with_output=None');f.write_text(s)
 shutil.copyfile(OLD/'inputs/payload1.json',d/'payload8k.json');shutil.copyfile(OLD/'inputs/long-payload.json',d/'payload256k.json')
 import subprocess
 subprocess.run(['python3',str(d/'patch.py')],check=True)
for f in P.glob('*.py'):compile(f.read_bytes(),str(f),'exec')
print('CONTROL_AND_PROFILE_STAGED')
