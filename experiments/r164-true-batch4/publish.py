"""Package whitelisted evidence and stage documentation, never deploy code."""
import hashlib,json,shutil,subprocess,tarfile,time
from pathlib import Path
P=Path(__file__).resolve().parent;repo=Path('/root/qwen-ascend')
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
v=json.loads((P/'final-validation.json').read_text());assert v['status']=='passed'
assert not (P/'original-private.json').exists() and not (P/'lease.active').exists()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip(),'PRESERVE_UNRELATED_WORK'
for n in ('harness-manifest.json','nd-manifest.json'):
    assert all(sha(P/k)==h for k,h in json.loads((P/n).read_text()).items())
scripts=sorted(f.name for f in P.iterdir() if f.is_file() and f.suffix in ('.py','.sh'))
data=['analysis.json','authorization.json','window.json','nd-resume.json','nd-resume-error.json',
      'nd-manifest.json','nd-manifest-resume-error.json','harness-manifest.json','canonical-manifest.json',
      'selected-config.json','cpu-preflight.json','probe-device0.json','probe-device1.json','probe-device0-initial.json',
      'probe-nd.json','batch4-results.json','batch1-results.json','batch4-graph-check.json','batch1-graph-check.json',
      'batch4-long-graph-check.json','batch1-long-graph-check.json','batch4-nd-launch.json','batch1-nd-launch.json',
      'restore-complete.json','restored-health.json','final-validation.json','formal-smoke.json','idle-check.jsonl']
assert all((P/n).is_file() for n in data)
initial=sorted(str(f.relative_to(P)) for f in (P/'attempt-scatter').iterdir() if f.is_file())
logs=['controller.log','controller-nd.log','batch4.log','batch4-nd.log','batch1-nd.log','restore.log']
assert all((P/n).is_file() for n in logs)
# Preserve the exact failed continuation helper by its already recorded hash.
old=(P/'resume_nd.py').read_text()
old=old.replace("archive=P/'attempt-scatter';archive.mkdir(exist_ok=True)","archive=P/'attempt-scatter';assert not archive.exists();archive.mkdir()")
old=old.replace("if f.exists() and not (archive/n).exists():shutil.copy2(f,archive/n)","if f.exists():shutil.copy2(f,archive/n)")
begin=old.index("    load=subprocess.check_output(['systemctl','show','fnx-r164-safety.service'")
end=old.index("    subprocess.run(['systemd-run','--unit=fnx-r164-run-nd'",begin)
old=old[:begin]+"    subprocess.run(['systemctl','restart','fnx-r164-safety.service'],check=True)\n"+old[end:]
failed=P/'attempt-resume-error/resume_nd.py';failed.parent.mkdir(exist_ok=True);failed.write_text(old)
assert sha(failed)==json.loads((P/'nd-manifest-resume-error.json').read_text())['resume_nd.py']
initial+=['attempt-resume-error/resume_nd.py']
bench=repo/'benchmarks/r164-true-batch4';ops=repo/'experiments/r164-true-batch4';doc=repo/'docs/r164-true-batch4.md'
assert not bench.exists() and not ops.exists() and not doc.exists()
bench.mkdir();ops.mkdir()
for n in scripts:shutil.copy2(P/n,ops/n)
for n in data+initial:
    dst=bench/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(P/n,dst)
for n in ('batch4-nd.log','batch1-nd.log'):
    lines=[x for x in (P/n).read_text(errors='replace').splitlines() if len(x)<3000]
    (bench/(n+'.console.txt')).write_text('\n'.join(lines)+'\n')
shutil.copy2(P/'report.md',doc)
readme='''# r164: experimental true B4 full-model decode

This is the recorded private fixed-batch measurement, not a deployed HTTP
continuous-batching scheduler. Production remains r152. Full quality testing
has not been completed for the changed numerical path.

The frozen initial harness and the supplemental ND writer are both retained.
`prepare_nd.py` generates readable `batch_ops_nd.py` and `bench_nd.py` from the
initial sources. The ND revision keeps three short cohorts and one long cohort,
respects both stop tokens, and uses chunk2048 for shared-prefix prefill.

The measured runtime was a private copy of the 45 canonical production files;
`prepare_runtime.py` enables the explicitly experimental B dimension guard.
The immutable source/flag checksums and complete result timelines are in
`../../benchmarks/r164-true-batch4/`. Recreate the private `runtime/fnx` and
`selected-config.json` in a NEW diagnostic directory, check the canonical hashes,
and obtain a NEW bounded exclusive window before any full model is loaded.
The paths in the recovery/controller scripts describe this particular run and
must be reviewed for a future machine/run. Never reuse the old authorization or
reset its deadline; the archived start marker deliberately prevents that.

The initial 256k scatter graph OOM and continuation-guard startup error are
preserved. Cross-device old-graph reset emitted EE1010 warnings; subsequent
fresh graph/eager checks passed, but the lifecycle issue remains a limitation.
No production engine, server scheduler, launcher, or model weights are changed.

See `../../docs/r164-true-batch4.md` for results, definitions and recovery proof.
'''
(ops/'README.md').write_text(readme)
base=repo/'README.md';prior=base.read_bytes();marker=b'<!-- r164-true-batch4 -->';assert marker not in prior
prefix=('<!-- r164-true-batch4 -->\n'
'> 当前生产保持 **r152**。2026-11-14 真正固定B4整模型实验：8k四路合计 **33.22 token/s**、每路 **8.31**；约256k合计 **20.46**、每路 **5.12**。实验路径未部署，正式HTTP调度未改；同路径B1与生产参考分开报告。[完整结果、边界及恢复](docs/r164-true-batch4.md)。\n'
'> 以下保留历史版本记录，当前状态以上述实测与恢复记录为准。\n'
'<!-- /r164-true-batch4 -->\n\n').encode()
base.write_bytes(prefix+prior);assert base.read_bytes()[len(prefix):]==prior
public=sorted(set(scripts+data+initial+logs+['report.md','source.tgz']))
public+=sorted(n for n in json.loads((P/'harness-manifest.json').read_text()) if n.startswith('runtime/'))
assert 'original-private.json' not in public
receipt=dict(epoch=time.time(),status='completed',production_revision='r152',candidate_deployed=False,
    base_git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
    files={n:sha(P/n) for n in public},readme_previous_sha256=hashlib.sha256(prior).hexdigest(),
    readme_history_preserved=True,raw_logs_in_archive=True,private_environment_excluded=True)
(P/'PUBLICATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2))
shutil.copy2(P/'PUBLICATION.json',bench/'PUBLICATION.json')
with tarfile.open(P/'evidence.tgz','w:gz') as tar:
    for n in public+['PUBLICATION.json']:tar.add(P/n,arcname=n,recursive=False)
(P/'evidence.sha256').write_text(sha(P/'evidence.tgz')+'  evidence.tgz\n')
files=sorted(f for root in (bench,ops) for f in root.rglob('*') if f.is_file())
(bench/'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.relative_to(repo)}\n' for f in files))
print('PUBLISHED_LAB_FILES',len(files),'ARCHIVE_FILES',len(public)+1,'ARCHIVE_SHA256',sha(P/'evidence.tgz'))
