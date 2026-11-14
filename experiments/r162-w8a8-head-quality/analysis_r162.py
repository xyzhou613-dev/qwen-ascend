"""Read-only paired analysis against the acceptance policy frozen before evaluation.

Incomplete files produce an incomplete decision, never a release recommendation.
No task text or generated answer is included in the analysis artifact.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path


def read_json(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def rows(path, key):
    out = {}
    if not path.exists():
        return out
    for line in path.read_bytes().splitlines(keepends=True):
        if not line.endswith(b'\n'):
            continue  # Writer may still be appending; do not repair its file.
        row = json.loads(line)
        assert row[key] not in out, ('duplicate checkpoint key', path.name, row[key])
        out[row[key]] = row
    return out


def binomial_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n or p == 0:
        return 1.0
    if p == 1:
        return 0.0
    terms = [math.lgamma(n+1)-math.lgamma(j+1)-math.lgamma(n-j+1)
             + j*math.log(p)+(n-j)*math.log1p(-p) for j in range(k+1)]
    m = max(terms)
    return min(1.0, math.exp(m)*math.fsum(math.exp(t-m) for t in terms))


def cp_bound(k, n, alpha, upper):
    assert 0 <= k <= n and n > 0 and 0 < alpha < 1
    if upper and k == n:
        return 1.0
    if not upper and k == 0:
        return 0.0
    lo, hi = 0.0, 1.0
    target = alpha if upper else 1-alpha
    count = k if upper else k-1
    for _ in range(65):
        mid = (lo+hi)/2
        if binomial_cdf(count, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo+hi)/2


def paired_task_stats(pairs, confidence, margin):
    n = len(pairs)
    assert n
    improved = sum(not b and c for b, c in pairs)
    worsened = sum(b and not c for b, c in pairs)
    alpha = 1-confidence
    # When improvements are zero their lower bound is deterministically zero;
    # otherwise split alpha to cover both estimated bounds by a union bound.
    tail = alpha/2 if improved else alpha
    lower_i = cp_bound(improved, n, tail, False) if improved else 0.0
    upper_w = cp_bound(worsened, n, tail, True)
    lower = lower_i-upper_w
    return dict(n=n, baseline_correct=sum(b for b, c in pairs),
                candidate_correct=sum(c for b, c in pairs), improved=improved,
                worsened=worsened, delta=(improved-worsened)/n,
                conservative_one_sided_lower=lower,
                confidence=confidence, margin=margin, pass_margin=lower >= -margin)


def paired_losses(candidate, baseline):
    paired = []
    for key in sorted(candidate.keys() & baseline.keys(), key=str):
        c, b = candidate[key], baseline[key]
        assert c['ids_sha256'] == b['ids_sha256'], ('input mismatch', key)
        assert c['scored'] == b['scored'] == len(c['nll_tokens']) == len(b['nll_tokens'])
        assert c['mode'] == b['mode'], ('execution mode mismatch', key)
        assert all(math.isfinite(x) for r in (b, c) for x in r['nll_tokens'])
        paired.append(dict(key=key, scored=c['scored'],
                           baseline_nll=math.fsum(b['nll_tokens'])/b['scored'],
                           candidate_nll=math.fsum(c['nll_tokens'])/c['scored'],
                           delta=math.fsum(x-y for x, y in zip(c['nll_tokens'], b['nll_tokens']))/c['scored']))
    return paired


def weighted_loss(pairs):
    total = sum(r['scored'] for r in pairs)
    if not total:
        return dict(pairs=0, scored=0)
    return dict(pairs=len(pairs), scored=total,
                baseline_nll=math.fsum(r['baseline_nll']*r['scored'] for r in pairs)/total,
                candidate_nll=math.fsum(r['candidate_nll']*r['scored'] for r in pairs)/total,
                delta=math.fsum(r['delta']*r['scored'] for r in pairs)/total)


def http_validation(p):
    records, crosscheck, unavailable = {}, [], []
    for phase in ('baseline', 'smooth', 'combined', 'rollback'):
        records[phase] = {}
        for name in ('8k.json', '256k.json', 'cached256k.json', 'sustained1024.json'):
            data = read_json(p/phase/name)
            if data is None:
                continue
            entries = data if isinstance(data, list) else [data]
            records[phase][name] = entries
            for i, row in enumerate(entries):
                if 'ttft_s' not in row:
                    assert name == 'cached256k.json'
                    unavailable.append(dict(phase=phase, file=name, index=i,
                                            reason='non-stream cache measurement has no client first-text timestamp'))
                    continue
                count = row['stats']['tokens']
                assert count > 1
                client = 1000*(row['end_to_end_s']-row['ttft_s'])/(count-1)
                crosscheck.append(dict(phase=phase, file=name, index=i, tokens=count,
                                       n_prefill=row['stats']['n_prefill'], client_tpot_ms=client,
                                       stats_tpot_ms=row['stats']['tpot_ms'],
                                       delta_ms=client-row['stats']['tpot_ms']))
    frozen_input_match, fresh_execution = {}, {}
    aa_answers = {}
    for name, expected in (('8k.json', 8060), ('256k.json', 262012)):
        base = records['rollback'].get(name, [])
        fresh_execution[name] = {phase: dict(completed=len(values.get(name, [])), expected=3,
                                             all_prefilled=all(r['stats']['n_prefill'] == expected for r in values.get(name, [])),
                                             all_output_128=all(r['stats']['tokens'] == 128 for r in values.get(name, [])))
                                 for phase, values in records.items()}
        frozen_input_match[name] = {phase: all(row['payload_sha256'] == base[i]['payload_sha256']
                                               for i, row in enumerate(values.get(name, [])))
                                    for phase, values in records.items() if len(values.get(name, [])) == len(base) == 3}
        rolled = records['rollback'].get(name, [])
        aa_answers[name] = None  # r162 has one same-run r152 baseline; no A/A rerun
    return dict(fresh_execution=fresh_execution, frozen_input_match=frozen_input_match,
                baseline_rollback_answers_identical=aa_answers,
                client_stats_tpot_max_abs_delta_ms=max((abs(r['delta_ms']) for r in crosscheck), default=None),
                client_stats_tpot=crosscheck, client_stats_crosscheck_unavailable=unavailable)


def analyze(p):
    policy = read_json(p/'acceptance-policy.json')
    assert policy['quality_gate_requires_complete_pairs']
    nll_source = read_json(p/'private-quality/nll-inputs.json')['cases']
    task_raw = (p/'private-quality/tasks.jsonl').read_bytes()+(p/'quality-extra-tasks.jsonl').read_bytes()
    task_source = [json.loads(x) for x in task_raw.splitlines()]
    task_sha = hashlib.sha256(task_raw).hexdigest()
    references = read_json(p/'canonical-grader-summary.json', {})
    if references:
        assert references['input_sha256'] == task_sha
    reference_complete = references.get('completed') == references.get('expected') == policy['total_tasks']
    reference_passed = reference_complete and references.get('all_references_passed') and references.get('negative_format_checks_passed')
    task_ids = {r['id'] for r in task_source}
    assert len(task_ids) == len(task_source) == policy['total_tasks']
    natural_source = read_json(p/'private-natural/inputs.json')
    c_nll, b_nll = rows(p/'combined-nll.jsonl', 'case'), rows(p/'rollback-nll.jsonl', 'case')
    for phase, records in (('combined', c_nll), ('rollback', b_nll)):
        for i, row in records.items():
            assert row['variant'] == phase
            assert row['ids_sha256'] == nll_source[i]['ids_sha256']
            assert row['domain'] == nll_source[i]['domain']
            assert row['mode'] == 'decode' and not row['legacy']
            assert row['warm'] == len(nll_source[i]['ids'])-1-row['scored']
    nll_pairs = paired_losses(c_nll, b_nll)
    nll = weighted_loss(nll_pairs)
    nll['completed_candidate_cases'] = len(c_nll)
    nll['completed_baseline_cases'] = len(b_nll)
    slices = {}
    domains = sorted({r['domain'] for r in nll_source})
    for domain in domains:
        indices = [i for i, r in enumerate(nll_source) if r['domain'] == domain]
        assert len(indices) == 49
        for j in range(7):
            selected = set(indices[j*7:(j+1)*7])
            s = weighted_loss([r for r in nll_pairs if r['key'] in selected])
            s['complete'] = s['pairs'] == 7
            s['pass_limit'] = s['complete'] and s.get('delta', math.inf) <= policy['slice_nll_limit']
            slices[f'{domain}/{j}'] = s
    nll['slices'] = slices
    nll['complete'] = len(nll_pairs) == len(nll_source) == 196 and nll['scored'] == 100352
    nll['pass_limit'] = (nll['complete'] and nll['delta'] <= policy['mean_nll_limit']
                         and all(s['pass_limit'] for s in slices.values()))

    c_nat, b_nat = rows(p/'combined-natural.jsonl', 'case'), rows(p/'rollback-natural.jsonl', 'case')
    for phase, records in (('combined', c_nat), ('rollback', b_nat)):
        for i, row in records.items():
            assert row['variant'] == phase
            assert row['ids_sha256'] == natural_source[i]['ids_sha256']
            assert row['mode'] == 'prefill' and not row['legacy']
            assert row['warm'] == len(natural_source[i]['ids'])-1-row['scored']
    nat_pairs = paired_losses(c_nat, b_nat)
    natural = weighted_loss(nat_pairs)
    natural['completed_candidate_cases'] = len(c_nat)
    natural['completed_baseline_cases'] = len(b_nat)
    natural['slices'] = [dict(name=natural_source[r['key']]['name'], **r) for r in nat_pairs]
    natural['complete'] = len(nat_pairs) == len(natural_source) == 10 and natural['scored'] == 1990
    natural['pass_limit'] = (natural['complete'] and
                             all(r['delta'] <= policy['natural_prefill_slice_nll_limit'] for r in nat_pairs))
    c_old = read_json(p/'combined-prefill-quality.json', {})
    b_old = read_json(p/'rollback-prefill-quality.json', {})
    old_pairs = paired_losses(c_old, b_old)
    assert all(r['mode'] == 'prefill' and not r['legacy'] for records in (c_old, b_old) for r in records.values())
    regression = weighted_loss(old_pairs)
    regression['slices'] = old_pairs
    regression['complete'] = len(old_pairs) == 5 and regression['scored'] == 995
    regression['pass_limit'] = (regression['complete'] and
                                all(r['delta'] <= policy['natural_prefill_slice_nll_limit'] for r in old_pairs))
    s_old = read_json(p/'smooth-prefill-quality.json', {})
    smooth_old_pairs = paired_losses(s_old, b_old)
    regression['smooth_control'] = weighted_loss(smooth_old_pairs)
    regression['smooth_control']['slices'] = smooth_old_pairs
    regression['smooth_control']['complete'] = len(smooth_old_pairs) == 5
    regression['smooth_control']['all_slices_within_limit'] = (len(smooth_old_pairs) == 5 and
        all(r['delta'] <= policy['natural_prefill_slice_nll_limit'] for r in smooth_old_pairs))

    c_task, b_task = rows(p/'quality-combined.jsonl', 'id'), rows(p/'quality-rollback.jsonl', 'id')
    for phase, records in (('combined', c_task), ('rollback', b_task)):
        meta = read_json(p/f'quality-{phase}-meta.json')
        assert set(records).issubset(task_ids)
        assert all(r['input_sha256'] == task_sha for r in records.values())
        assert not records or (meta and meta['input_sha256'] == task_sha
                               and meta['health_settings']['diagnostic_variant'] == phase)
        assert all(r['config_sha256'] == meta['config_sha256'] and r['url'] == meta['url']
                   for r in records.values())
    pairs = []
    domain_map = {'cmmlu': 'cmmlu', 'mmlu-pro': 'mmlu_pro', 'gsm8k': 'math',
                  'humaneval': 'code', 'mbpp': 'code', 'structure': 'json'}
    for task in task_source:
        key = task['id']
        if key not in c_task or key not in b_task:
            continue
        root = key.split('/')[0]
        domain = domain_map.get(root)
        if domain is None and task['kind'] == 'json':
            domain = 'json'
        assert domain in policy['task_domains'], (key, domain)
        pairs.append(dict(id=key, domain=domain, benchmark=root,
                          b=bool(b_task[key]['result']['correct']), c=bool(c_task[key]['result']['correct'])))
    task_domains = {}
    for domain in policy['task_domains']:
        subset = [(r['b'], r['c']) for r in pairs if r['domain'] == domain]
        expected = 300 if domain in ('cmmlu', 'mmlu_pro') else 200
        d = paired_task_stats(subset, policy['task_confidence'], policy['task_margin']) if subset else dict(n=0, pass_margin=False)
        d['expected'] = expected
        d['complete'] = len(subset) == expected
        d['pass_margin'] &= d['complete']
        selected = [r for r in pairs if r['domain'] == domain]
        d['baseline_truncated'] = sum(b_task[r['id']].get('finish_reason') == 'length' for r in selected)
        d['candidate_truncated'] = sum(c_task[r['id']].get('finish_reason') == 'length' for r in selected)
        task_domains[domain] = d
    sub = {}
    for name in sorted({r['benchmark'] for r in pairs}):
        subset = [(r['b'], r['c']) for r in pairs if r['benchmark'] == name]
        delta = sum(int(c)-int(b) for b, c in subset)/len(subset)
        sub[name] = dict(n=len(subset), baseline_correct=sum(b for b, c in subset),
                         candidate_correct=sum(c for b, c in subset), delta=delta,
                         regression_review=delta < -policy['subbenchmark_point_regression_review_threshold'])
    tasks = dict(paired=len(pairs), expected=len(task_source), domains=task_domains,
                 completed_candidate=len(c_task), completed_baseline=len(b_task),
                 subbenchmarks=sub, complete=len(pairs) == len(task_source))
    tasks['pass_margin'] = tasks['complete'] and all(d['pass_margin'] for d in task_domains.values())
    tasks['requires_regression_review'] = any(d['regression_review'] for d in sub.values())

    parity = {}
    for phase in ('combined', 'rollback'):
        r = read_json(p/f'{phase}-scoring-parity.json')
        parity[phase] = dict(complete=r is not None,
                             pass_limit=bool(r is not None and r['max_abs_delta'] < .00005),
                             max_abs_delta=r['max_abs_delta'] if r else None)
        if phase == 'rollback' and r:
            parity[phase]['legacy_reference_pass'] = abs(r['legacy']['mean_nll']-5.178334) < .00001
            parity[phase]['pass_limit'] &= parity[phase]['legacy_reference_pass']

    performance = {phase: read_json(p/phase/'summary.json')
                   for phase in ('baseline', 'smooth', 'combined', 'rollback')}
    changes = {}
    metrics = ('median_8k_tpot_ms', 'median_256k_tpot_ms', 'cached_256k_tpot_ms',
               'sustained1024_tpot_ms', 'median_8k_ttft_s', 'median_256k_ttft_s')
    for phase in ('smooth', 'combined'):
        if not performance[phase]:
            continue
        changes[phase] = {}
        for reference in ('baseline', 'rollback'):
            if performance[reference]:
                changes[phase][reference] = {
                    k: dict(candidate=performance[phase][k], baseline=performance[reference][k],
                            reduction_percent=100*(1-performance[phase][k]/performance[reference][k]))
                    for k in metrics}
    complete = all(r['complete'] for r in (nll, natural, regression, tasks)) and all(r['complete'] for r in parity.values()) and reference_complete
    passed = complete and reference_passed and all(r['pass_limit'] for r in (nll, natural, regression)) and tasks['pass_margin'] and not tasks['requires_regression_review'] and all(r['pass_limit'] for r in parity.values())
    status = ('invalid_reference_grading' if reference_complete and not reference_passed else
              'paired_quality_pass' if passed else 'paired_quality_fail' if complete else 'incomplete_no_acceptance')
    return dict(status=status,
                complete=complete, passed=passed, task_input_sha256=task_sha,
                provenance_sha256={str(f.relative_to(p)): hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in (p/'acceptance-policy.json', p/'private-quality/nll-inputs.json',
                                             p/'private-natural/inputs.json', p/'runtime-code-manifest.json',
                                             p/'harness-manifest.json')},
                policy=policy, reference_grading=references, nll=nll, fresh_natural_prefill=natural,
                legacy_natural_regression=regression, tasks=tasks,
                scoring_parity=parity, performance=performance, performance_changes=changes,
                measurement_validation=http_validation(p),
                limits=['Frozen public benchmarks may appear in model pretraining.',
                        'JSON extraction does not establish native tool-call protocol quality.',
                        'Literary prefill NLL does not establish comprehensive long-context QA accuracy.'])


def self_test():
    assert abs(cp_bound(0, 200, .05, True)-(1-.05**(1/200))) < 1e-12
    assert abs(cp_bound(200, 200, .05, False)-.05**(1/200)) < 1e-12
    assert paired_task_stats([(True, True)]*200, .95, .02)['pass_margin']
    assert not paired_task_stats([(True, True)]*100, .95, .02)['pass_margin']
    assert not paired_task_stats([(True, False)]*20+[(True, True)]*180, .95, .02)['pass_margin']
    # CP bounds are complementary under exchanging successes and failures.
    assert abs(cp_bound(7, 40, .025, True)+cp_bound(33, 40, .025, False)-1) < 1e-12
    print('PAIRED_ANALYSIS_SELFTEST_PASS')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path(__file__).resolve().parent)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.self_test:
        self_test()
    else:
        result = analyze(args.root)
        output = args.output or args.root/'paired-analysis.json'
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
        print(result['status'], 'NLL', result['nll']['pairs'], 'tasks', result['tasks']['paired'],
              'natural', result['fresh_natural_prefill']['pairs'])
