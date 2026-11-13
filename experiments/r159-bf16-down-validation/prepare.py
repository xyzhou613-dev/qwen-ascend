"""Prepare a private BF16-rounded down candidate; never stop or load a model."""
import hashlib
import json
import shutil
from pathlib import Path

P = Path(__file__).resolve().parent
R157 = Path('/models/fnx-diagnostics/r157-combined-quality-20261113')
R158 = Path('/models/fnx-diagnostics/r158-bf16-down-20261113')
LIVE = Path('/root/ascend-dev/fnx/fnx')
assert not (P/'baseline').exists(), 'REFUSED_RUNTIME_OVERWRITE'
for phase in ('baseline', 'combined'):
    dest = P/phase/'fnx'
    shutil.copytree(R157/phase/'fnx', dest, ignore=shutil.ignore_patterns('__pycache__'))
    if phase == 'combined':
        shutil.copyfile(R158/'libfnx_w4_bf16w.so', dest/'libfnx_w4_bf16w.so')
        f = dest/'kmajor_down.py'
        s = f.read_text()
        assert s.count('_lib=None') == 1 and s.count('rc=_lib.fnx_w4_kmajor(') == 1
        s = s.replace('_lib=None', '_lib=None\n_rounded_lib=None', 1)
        s = s.replace('global _lib\n', 'global _lib,_rounded_lib\n', 1)
        anchor = ' out=torch.empty((10,2560),device=x.device,dtype=torch.bfloat16)'
        assert s.count(anchor) == 1
        load = ''' if _rounded_lib is None:
  _rounded_lib=ctypes.CDLL(str(Path(__file__).with_name('libfnx_w4_bf16w.so')))
  _rounded_lib.fnx_w4_bf16w.argtypes=[ctypes.c_void_p]*5+[ctypes.c_int,ctypes.c_void_p]
  _rounded_lib.fnx_w4_bf16w.restype=ctypes.c_int
'''
        s = s.replace(anchor, load+anchor, 1)
        s = s.replace('rc=_lib.fnx_w4_kmajor(', 'rc=_rounded_lib.fnx_w4_bf16w(', 1)
        s = s.replace(',256,torch.npu.current_stream', ',int(os.environ.get("FNX_BF16_DOWN_TILE","256")),torch.npu.current_stream', 1)
        f.write_text(s)
        audit=dest/'audit.py'
        a=audit.read_text()
        old='def status():return dict(pid=os.getpid(),layers=rows,kmajor_hits=hits)'
        assert a.count(old)==1
        a=a.replace(old,'''def status():
 from . import kmajor_down as D
 return dict(pid=os.getpid(),layers=rows,kmajor_hits=hits,
  rounded_library=getattr(D._rounded_lib,'_name',None),rounded_tile=os.environ.get('FNX_BF16_DOWN_TILE'))''',1)
        audit.write_text(a)
    for f in dest.glob('*.py'):
        compile(f.read_bytes(), str(f), 'exec')
for name in ('snapshot.py', 'idle_guard.py', 'check_ready.py', 'formal_smoke.py', 'quality_core.py'):
    shutil.copyfile(R157/name, P/name)
smoke=P/'formal_smoke.py'
s=smoke.read_text()
s=s.replace("ref=json.load(open(P/'baseline/8k.json'))[0]", "ref_path=P/'baseline/8k.json'\nif not ref_path.exists():ref_path=Path('/models/fnx-diagnostics/r157-combined-quality-20261113/baseline/8k.json')\nref=json.load(open(ref_path))[0]",1)
smoke.write_text(s)
shutil.copytree(R157/'inputs', P/'inputs')
routes = (R157/'qa_routes.py').read_text()
routes = routes.replace("P/'private-natural/inputs.json'", "Path('/models/fnx-diagnostics/r157-combined-quality-20261113/private-natural/inputs.json')")
(P/'qa_routes.py').write_text(routes)
bench = (R157/'http_bench.py').read_text()
# Results remain comparable; every phase still uses the same frozen payload bytes.
(P/'http_bench.py').write_text(bench)
source = {phase: {f.name: hashlib.sha256(f.read_bytes()).hexdigest()
                 for f in (P/phase/'fnx').iterdir() if f.suffix in ('.py', '.so')}
          for phase in ('baseline', 'combined')}
(P/'runtime-code-manifest.json').write_text(json.dumps(source, indent=2)+'\n')
(P/'production-before-hashes.json').write_text(json.dumps({f.name: hashlib.sha256(f.read_bytes()).hexdigest()
    for f in LIVE.iterdir() if f.is_file() and f.suffix in ('.py', '.so')}, indent=2)+'\n')
shutil.copyfile('/root/ascend-dev/p275.sh', P/'p275-before.sh')
policy = dict(scope='bounded debugging, not full quality acceptance',
    component_modes=['production', 'old', 'rounded128', 'rounded256'],
    component_order='alternating forward/reverse per round',
    component_banks=64, component_rounds=16,
    component_gate='best rounded tile must save >=2 percent versus production on each card; select using down+epilogue only',
    layer_indices=[0, 3, 24, 27], layer_error_limit=.001,
    legacy_prefill_slice_limit=.05, baseline_repeat_nll_tolerance=.00005,
    nll_screen_cases=[0,49,98,147], nll_screen_acceptance=False,
    model_quality_gate='r157 policy remains unchanged; this short run cannot establish full acceptance',
    max_window_seconds=3600, restore_trigger_seconds=2400, backup_trigger_seconds=2580,
    recovery_reserve_seconds=1200, input_sha256={f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in (P/'inputs').glob('*.json')},
    source_r158_library_sha256=hashlib.sha256((R158/'libfnx_w4_bf16w.so').read_bytes()).hexdigest())
assert policy['source_r158_library_sha256']=='7861f2c3128a0ff25989f90c4c5f19938e361506d2b72e6de52b30e41085eda9'
(P/'debug-policy.json').write_text(json.dumps(policy, indent=2)+'\n')
print('PRIVATE_RUNTIMES_PREPARED_NO_MODEL_STARTED')
