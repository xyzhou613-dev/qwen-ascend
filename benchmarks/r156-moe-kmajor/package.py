"""Explicit public evidence packaging; excludes datasets, weights and Docker backups."""
import difflib,hashlib,json,pathlib,shutil,tarfile
P=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/root/qwen-ascend')
assert json.load(open(P/'final-operations.json'))['recovered_http_verified']
B=R/'benchmarks/r156-moe-kmajor';X=R/'experiments/r156-moe-kmajor'
assert not B.exists() and not X.exists(),'refuse overwrite existing work'
B.mkdir();X.mkdir()
sources=['w4_kmajor.cpp','diag_epilogue.cpp','fused_down.cpp','build.sh','kmajor_down.py','libfnx_w4_kmajor.so']
for name in sources:shutil.copyfile(P/name,X/name)
for f in P.iterdir():
 if f.is_file() and f.suffix in ('.json','.jsonl','.log','.txt','.py','.sh','.ps1') and f.name not in sources:
  shutil.copyfile(f,B/f.name)
shutil.copyfile(P/'quality-sources/source-manifest.json',B/'quality-source-manifest.json')
base=pathlib.Path('/root/ascend-dev/fnx/fnx/engine.py');candidate=P/'candidate/fnx/engine.py'
(X/'selected-engine.patch').write_text(''.join(difflib.unified_diff(base.read_text().splitlines(True),candidate.read_text().splitlines(True),fromfile='a/fnx/engine.py',tofile='b/fnx/engine.py')))
(X/'README.md').write_text('''# r156 original-layout MoE down experiment

Research candidate only. Production remains r152; no whole-model TPOT or final quality acceptance for this candidate.

The selected implementation is the two-kernel path: direct K-major down + diagonal epilogue. `fused_down.cpp` is a rejected slower one-kernel variant. Its source and measured results are retained; `kmajor_down.merge` does not call it. `diag_epilogue.cpp` preserves the production epilogue rounding and sum order.

Build `bash build.sh` inside the existing Ascend CANN 9.1.0 environment. This uses a single compiler translation unit to avoid duplicate g_tilingKey symbols. The library ABI is for dav-c220-vec and the measured 910B4-1 environment.

The combined selected-engine.patch is against production r152. It includes the prior SmoothQuant constructor/dispatch changes plus the new decode-only MoE branch. Candidate packaging also needs `fnx/w8a8.py` and `fnx/smooth_w8a8.py` from experiments/r154-w8a8-gemv, and this module/library alongside them. Enable the prior SmoothQuant environment with its calibration directory, then explicitly set FNX_KMAJOR_DOWN=1. Defaults preserve the original path. This is not a production deployment instruction or quality approval.

Before the new MoE path, the activation and FP32 SwiGLU remain unchanged. The new path consumes cached BF16 scales and leaves all grouped prefill paths untouched. Complete prefill/decode and model quality remain to be validated. The abandoned install/apply_diag interception was bypassed by the production fused branch; the explicit selected-engine.patch now targets that actual branch.
''')
(B/'README.md').write_text('''# r156 evidence

See [report](../../docs/r156-moe-kmajor.md). Real component/layer timings were collected before the 17:02 host reboot. They are not new HTTP TPOT values. After reboot, physical node indices changed and the original container could not start; recovery-device-change.json and final-operations.json record the authorized restoration. Production source and p275 hashes were unchanged. Rebuild performance baselines before further full-model comparisons.

`probe.py` tests the down projection including the legacy standalone diagonal extraction baseline. `merge_probe.py` measures the correct production fused baseline versus selected split and rejected fused candidates. `layer_probe.py` compares SmoothQuant alone versus the combination on four real layers. Every tested layer's dispatch counter increased by 33; see dispatch-audit.json. The wrapper now additionally asserts each layer's 16 pre-timing calls individually. The candidate library/module is in experiments/r156-moe-kmajor.

`fetch_quality.ps1`, freeze_quality.py and freeze_nll.py prepare immutable held-out inputs. Inputs contain 1100 tasks and 100352 scored NLL targets; no candidate quality run has completed. Official source revisions and SHA hashes are recorded. Private dataset snapshots, question/answer text, canonical code/tests and token IDs are retained on the server, not redistributed here. Re-fetch the pinned sources to reproduce. The source pool for MMLU-Pro covers 12 categories, not its entire official test set.

`quality_eval.py` resumes frozen input hashes and grades code only in a network-disabled/read-only/nonroot resource-limited temporary CPU container. Framework selftests passed. quality-r152screen contains only 32 MCQ/JSON baseline questions, not the full suite. The 100k NLL input is benchmark knowledge/math/code text; natural-text and true long-prefill acceptance remain outstanding.

`recover.sh` is a site-specific historical recovery script. It stops the Docker daemon only after confirming that every container is stopped, backs up the existing container configuration privately, and changes only the two missing device mappings. It does not recreate the container. Do not treat it as an ordinary experiment launcher. Docker raw configs and credential-bearing helpers are excluded.

No model weights, calibration vectors, hidden states, complete process ENV, graph caches, dataset copies or Docker raw configuration backups are included.
''')
shutil.copyfile(P/'report.md',R/'docs/r156-moe-kmajor.md')
doc=R/'docs/Qwen3.8-Flash-Next-2x910B4-技术文档.md';old=doc.read_bytes();assert b'## r156 ' not in old[:1200]
doc.write_bytes((P/'techdoc-block.md').read_bytes()+old)
for d in (B,X):
 files=sorted(f for f in d.rglob('*') if f.is_file() and f.name!='SHA256SUMS');(d/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(d))+'\n' for f in files))
archive=P/'r156-evidence.tar.gz'
with tarfile.open(archive,'w:gz') as tf:
 tf.add(B,arcname='r156-evidence/benchmarks');tf.add(X,arcname='r156-evidence/experiments');tf.add(P/'report.md',arcname='r156-evidence/report.md')
print('PACKAGED',hashlib.sha256(archive.read_bytes()).hexdigest(),archive.stat().st_size)
