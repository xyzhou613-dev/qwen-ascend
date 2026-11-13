"""Allowlist public evidence; never traverse private data/model/runtime trees."""
import hashlib
import json
import shutil
import tarfile
from pathlib import Path

P = Path(__file__).resolve().parent
REPO = Path('/root/qwen-ascend')
EXP = REPO/'experiments/r157-combined-quality'
BENCH = REPO/'benchmarks/r157-combined-quality'
OPS = REPO/'ops/fnx-boot-restore'
SOURCES = (
    'prepare.py', 'prepare_control.py', 'audit.py', 'check_ready.py', 'checkpoint.py',
    'controller.sh', 'launch.py', 'qa_routes.py', 'quality_client.py', 'quality_core.py',
    'quality_eval.py', 'freeze_natural.py', 'freeze_supplement.py', 'test_alignment.py',
    'test_checkpoint.py', 'progress.py', 'safety.sh', 'rescue.sh', 'snapshot.py',
    'stop_owned.py', 'idle_guard.py', 'http_bench.py', 'formal_smoke.py',
    'launch_profile.py', 'profile.sh', 'analyze_profile.py', 'analyze.py',
    'final_verify.py', 'package.py', 'report.py', 'README.md', 'followup-notes.md', 'validate_references.py',
    'dispatch_restore.py', 'dispatch-restore.service.conf', 'restore-dispatch.timer.conf',
    'rescue-deadline.timer.conf', 'finalize_documents.py')
ROOT_DATA = (
    'acceptance-policy.json', 'natural-freeze.json', 'runtime-code-manifest.json', 'quality-harness-manifest.json',
    'production-before-hashes.json', 'paired-analysis.json', 'final-validation.json', 'report.md',
    'restored-health.json', 'production-http.json', 'quality-harness-selftest.json',
    'boot-recovery-validation.json', 'a8w4-interface-inventory.json', 'combined-npu-snapshot.txt', 'judge-image.json',
    'canonical-grader.jsonl', 'canonical-grader-summary.json', 'topology.txt', 'topology-affinity.json',
    'combined-nll-integrity.json', 'rollback-nll-integrity.json',
    'ownership.json', 'controller.log', 'window-start.txt', 'window-policy.json', 'input-manifest.json',
    'restore-dispatch.json', 'controller-unit-result.json', 'cap-cancellation.json', 'dispatch-preflight.json',
    'technical-doc-update.json', 'techdoc-block.md')
OPS_FILES = ('boot_restore.sh', 'reconcile_devices.py', 'test_reconcile.py', 'fnx-boot-restore.service')


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)


def stripped_losses(value):
    if isinstance(value, dict):
        return {k: stripped_losses(v) for k, v in value.items() if k != 'argmax'}
    if isinstance(value, list):
        return [stripped_losses(v) for v in value]
    return value


def complete_rows(path, interrupted):
    lines = path.read_bytes().splitlines(keepends=True)
    if lines and not lines[-1].endswith(b'\n'):
        interrupted[path.name] = len(lines[-1])
        lines = lines[:-1]
    return [json.loads(line) for line in lines]


def main():
    final = json.loads((P/'final-validation.json').read_text())
    assert final['private_env_removed'] and final['experimental_processes_absent']
    assert not (P/'lease.active').exists()
    # A second packaging pass is deliberately rejected; inspect before replacing.
    assert not EXP.exists() and not BENCH.exists() and not OPS.exists()
    interrupted = {}
    for name in SOURCES:
        copy(P/name, EXP/name)
    for name in OPS_FILES:
        source = (Path('/etc/systemd/system') if name.endswith('.service')
                  else Path('/root/ascend-dev/fnx-boot-restore'))/name
        copy(source, OPS/name)
    copy(P/'README-boot.md', OPS/'README.md')
    source_inputs = {}
    for file in sorted((P/'inputs').glob('*.json')):
        source_inputs['inputs/'+file.name] = dict(bytes=file.stat().st_size,
                                                sha256=hashlib.sha256(file.read_bytes()).hexdigest())
    (P/'input-manifest.json').write_text(json.dumps(source_inputs, indent=2)+'\n')
    for name in ROOT_DATA:
        if (P/name).exists():
            copy(P/name, BENCH/name)
    for phase in ('baseline', 'smooth', 'combined', 'rollback'):
        for name in ('summary.json', '8k.json', '256k.json', 'cached256k.json', 'sustained1024.json'):
            if (P/phase/name).exists():
                copy(P/phase/name, BENCH/phase/name)
        for suffix in ('health.json', 'startup.json', 'runtime-audit.json', 'scoring-parity.json',
                       'prefill-quality.json'):
            source = P/f'{phase}-{suffix}'
            if source.exists():
                value = stripped_losses(json.loads(source.read_text()))
                target = BENCH/source.name
                target.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
        for kind in ('nll', 'natural'):
            source = P/f'{phase}-{kind}.jsonl'
            if source.exists():
                values = [stripped_losses(row) for row in complete_rows(source, interrupted)]
                (BENCH/source.name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in values))
    private_result_hashes = {}
    for phase in ('baseline', 'smooth', 'combined', 'rollback'):
        for suffix in ('nll.jsonl', 'natural.jsonl', 'scoring-parity.json', 'prefill-quality.json'):
            source = P/f'{phase}-{suffix}'
            if source.exists():
                private_result_hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    for phase in ('combined', 'rollback'):
        source = P/f'quality-{phase}.jsonl'
        if source.exists():
            private_result_hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
            redacted = []
            for row in complete_rows(source, interrupted):
                row.pop('answer', None)
                row['result'] = {k: v for k, v in row['result'].items() if k != 'parsed'}
                redacted.append(row)
            (BENCH/source.name).write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in redacted))
        for suffix in ('meta.json', 'summary.json'):
            source = P/f'quality-{phase}-{suffix}'
            if source.exists():
                copy(source, BENCH/source.name)
    (BENCH/'private-results-manifest.json').write_text(json.dumps(private_result_hashes, indent=2)+'\n')
    (BENCH/'public-result-export.json').write_text(json.dumps(dict(
        omitted_interrupted_tail_bytes=interrupted, raw_sources_preserved_on_server=True,
        generated_task_answers_omitted=True, argmax_vectors_omitted=True), indent=2)+'\n')
    for name in ('controller.log', 'scoring-mismatch.json', 'runtime-code-manifest.json', 'prepare-npu-score.py'):
        source = P/'initial-attempt'/name
        if source.exists():
            copy(source, BENCH/'initial-attempt'/name)
    for phase in ('combined', 'smooth'):
        root = P/('profile-'+phase)
        for name in ('profile_model.py', 'profile_support.py', 'patch.py'):
            copy(root/name, EXP/'profile-harness'/name)
        if not (root/'summary.json').exists():
            continue
        dest = BENCH/('profile-'+phase)
        for name in ('summary.json', 'runs.json', 'clock-alignment.json', '8k-output.json', '256k-output.json'):
            if (root/name).exists():
                copy(root/name, dest/name)
        for source in root.glob('*_host*.json'):
            copy(source, dest/source.name)
        archive = dest/'raw-kernel-task-csv.tgz'
        with tarfile.open(archive, 'w:gz') as tf:
            for name in ('kernel_details.csv', 'task_time.csv'):
                for source in sorted((root/'traces').rglob(name)):
                    tf.add(source, arcname=str(source.relative_to(root)))
        with tarfile.open(dest/'raw-system-json.tgz', 'w:gz') as tf:
            for source in sorted((root/'system').rglob('msprof_*.json')):
                tf.add(source, arcname=str(source.relative_to(root)))
        for source in (root/'aligned').glob('*.json'):
            with source.open('rb') as f, __import__('gzip').open(dest/(source.name+'.gz'), 'wb') as out:
                shutil.copyfileobj(f, out)
    for root in (EXP, BENCH, OPS):
        files = sorted(f for f in root.rglob('*') if f.is_file() and f.name != 'SHA256SUMS')
        (root/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(root))+'\n' for f in files))
    print('PUBLIC_EVIDENCE_PACKAGED', EXP, BENCH, OPS)


if __name__ == '__main__':
    main()
