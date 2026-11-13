"""Verify all frozen reference answers before interpreting model accuracy."""
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path
from checkpoint import append, rows
from quality_eval import grade

P = Path(__file__).resolve().parent


def main():
    started = time.time()
    assert os.environ['R157_RUN_ID'].startswith('canonical-')
    raw = (P/'private-quality/tasks.jsonl').read_bytes()+(P/'quality-extra-tasks.jsonl').read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    tasks = [json.loads(line) for line in raw.splitlines()]
    output = P/'canonical-grader.jsonl'
    done = {r['id']: r for r in rows(output)}
    assert all(r['input_sha256'] == digest for r in done.values())
    for task in tasks:
        if task['id'] in done:
            continue
        kind = task['kind']
        answer = (task['canonical_code'] if kind == 'code' else
                  json.dumps(task['expected']) if kind == 'json' else
                  '#### '+task['expected'] if kind == 'numeric' else task['expected'])
        result = grade(task, answer)
        row = dict(id=task['id'], kind=kind, input_sha256=digest,
                   result={k: v for k, v in result.items() if k != 'parsed'})
        append(output, row)
        done[task['id']] = row
        print('CANONICAL', row['id'], row['result']['correct'], flush=True)
    assert not grade(dict(kind='json', expected={'v': 1}), '{"v":true}')['correct']
    assert not grade(dict(kind='mcq', expected='A'), 'A or B')['correct']
    assert not grade(dict(kind='numeric', expected='1'), '#### nan')['correct']
    failures = [r['id'] for r in done.values() if not r['result']['correct']]
    summary = dict(completed=len(done), expected=len(tasks), input_sha256=digest,
                   code_references=sum(t['kind'] == 'code' for t in tasks),
                   failed_ids=failures, all_references_passed=not failures and len(done) == len(tasks),
                   negative_format_checks_passed=True, elapsed_s=time.time()-started,
                   no_model_inference=True)
    (P/'canonical-grader-summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print('ALL_REFERENCE_GRADING_COMPLETE', summary, flush=True)
    assert summary['all_references_passed'], failures


if __name__ == '__main__':
    def stop(*_):
        raise SystemExit(143)
    signal.signal(signal.SIGTERM, stop)
    try:
        main()
    finally:
        run_id = os.environ.get('R157_RUN_ID', '')
        if run_id.startswith('canonical-'):
            ids = subprocess.check_output(['docker', 'ps', '-q', '--filter', 'label=fnx.diagnostic=r157',
                                           '--filter', 'label=fnx.run='+run_id], text=True).split()
            if ids:
                subprocess.run(['docker', 'rm', '-f', *ids], check=True, timeout=20,
                               stdout=subprocess.DEVNULL)
