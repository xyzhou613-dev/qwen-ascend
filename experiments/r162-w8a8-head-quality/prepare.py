"""Prepare a head-A8W8 full-model candidate and immutable baseline continuation."""
import hashlib,json,shutil,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent
OLD=Path('/models/fnx-diagnostics/r157-combined-quality-20261113')
R161=Path('/models/fnx-diagnostics/r161-quant-v4-20261114')
assert not (P/'baseline').exists()
for phase,source in (('baseline','baseline'),('smooth','smooth'),('combined','smooth')):
    dest=P/phase/'fnx';shutil.copytree(OLD/source/'fnx',dest,ignore=shutil.ignore_patterns('__pycache__'))
    if phase=='combined':
        assert 'if callable(w):' in (dest/'engine.py').read_text()
        shutil.copyfile(R161/'quant_head.py',dest/'quant_head.py')
        f=dest/'model.py';s=f.read_text();anchor='        self.mix_norm = mx["hc_norm.weight"]'
        assert s.count(anchor)==1
        s=s.replace(anchor,'''        if with_head and os.environ.get("FNX_HEAD_A8W8", "0") == "1":
            from .quant_head import QuantHead
            from .audit import register_head
            self.lm_head = QuantHead.from_grouped(self.lm_head, lambda: None)
            register_head(self.lm_head)
            lm_head = None  # release the original grouped head before loading layers
'''+anchor,1);f.write_text(s)
        f=dest/'audit.py';s=f.read_text()
        anchor='def status():return dict(pid=os.getpid(),layers=rows,kmajor_hits=hits)'
        assert s.count(anchor)==1
        s=s.replace(anchor,'''head_rows=[]
head_calls={}
def register_head(head):
 row=dict(type=type(head).__name__,N=head.N,K=head.K,weight_fit_relative_l2=head.weight_fit_relative_l2,
  storage_bytes=head.weight_storage_bytes,chunks=[dict(weight_shape=list(w.shape),weight_dtype=str(w.dtype),
  scale_dtype=str(s.dtype),scale_shape=list(s.shape),first=a,last=b) for w,s,a,b in head.chunks])
 head_rows.append(row)
 original=type(head).__call__
 def tracked(obj,x):
  key=str(x.shape[0]);head_calls[key]=head_calls.get(key,0)+1
  return original(obj,x)
 type(head).__call__=tracked
def status():return dict(pid=os.getpid(),layers=rows,kmajor_hits=hits,heads=head_rows,head_calls=head_calls)''',1)
        f.write_text(s)
    for f in dest.glob('*.py'):compile(f.read_bytes(),str(f),'exec')
for name in ('snapshot.py','idle_guard.py','quality_client.py','quality_core.py','quality_eval.py',
             'checkpoint.py','qa_routes.py','http_bench.py','validate_references.py','stop_owned.py'):
    shutil.copyfile(OLD/name,P/name)
for name in ('inputs','private-natural'):
    shutil.copytree(OLD/name,P/name)
shutil.copytree(OLD/'private-quality',P/'private-quality')
for name in ('quality-extra-tasks.jsonl','acceptance-policy.json','natural-freeze.json','judge-image.json'):
    shutil.copyfile(OLD/name,P/name)
shutil.copyfile('/root/ascend-dev/p275.sh',P/'p275-before.sh')
before={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in Path('/root/ascend-dev/fnx/fnx').iterdir() if f.is_file() and f.suffix in ('.py','.so')}
assert before==json.loads((OLD/'production-before-hashes.json').read_text()),'R152_CHANGED_NO_BASELINE_REUSE'
(P/'production-before-hashes.json').write_text(json.dumps(before,indent=2)+'\n')
reused={}
for name in ('rollback-nll.jsonl','rollback-natural.jsonl','rollback-prefill-quality.json',
             'quality-rollback.jsonl','quality-rollback-meta.json'):
    raw=(OLD/name).read_bytes();target=raw;omitted=0
    if name.endswith('.jsonl'):
        lines=raw.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b'\n'):
            try:json.loads(lines[-1]);lines[-1]+=b'\n'
            except (json.JSONDecodeError,UnicodeDecodeError):omitted=len(lines[-1]);lines.pop()
        for line in lines:json.loads(line)
        target=b''.join(lines)
    (P/name).write_bytes(target)
    reused[name]=dict(source=str(OLD/name),source_sha256=hashlib.sha256(raw).hexdigest(),
        copied_sha256=hashlib.sha256(target).hexdigest(),omitted_interrupted_tail_bytes=omitted,source_untouched=True)
(P/'baseline-reuse-provenance.json').write_text(json.dumps(reused,indent=2)+'\n')
assert sum(1 for _ in (P/'quality-rollback.jsonl').read_bytes().splitlines())==999
runtime={phase:{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/phase/'fnx').iterdir()
               if f.is_file() and f.suffix in ('.py','.so')} for phase in ('baseline','smooth','combined')}
(P/'runtime-code-manifest.json').write_text(json.dumps(runtime,indent=2)+'\n')
policy=dict(max_seconds=21600,restore_after_seconds=20100,backup_after_seconds=20280,
    baseline_reuse_requires_unchanged_production=True,baseline_nll_repeat_tolerance=.00005,
    stage_order=['rollback r152 + remaining201 tasks','smooth W8A8 control','combined W8A8+headA8W8'],
    head_incremental_min_saved_ms=.5,minimum_output_speed_gain_fraction=.14,
    performance_gate_scenarios=['fresh8k','fresh262016'],short_prefill_slice_nll_limit=.05,
    full_quality_policy_sha256=hashlib.sha256((P/'acceptance-policy.json').read_bytes()).hexdigest(),
    complete_pairs_required=True,deployment_requires_complete_quality_and_safe_rollback=True,
    rollout_not_part_of_controller=True,
    stop_on_failed_short_gate=True,timeout_restore_with_checkpoints=True)
(P/'run-policy.json').write_text(json.dumps(policy,indent=2)+'\n')
print('R162_PREPARED_NO_ENGINE_NO_INTERRUPTION; BASELINE999_CHECKPOINTS_PRESERVED')
