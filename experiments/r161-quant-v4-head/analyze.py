"""Summarize measured components without treating proxies as HTTP TPOT."""
import json,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
load=lambda k,d:json.loads((P/f'{k}-device{d}.json').read_text())
layers=[r for d in (0,1) for r in load('layers',d)['records']]
abl=[r for d in (0,1) for r in load('ablation',d)['records']]
roll=[r for d in (0,1) for r in load('rollback',d)['records']]
def weighted(records,value):
    return sum(count*statistics.mean(value(r) for r in records if r['kind']==kind) for kind,count in (('linear',36),('full',12)))
assert sorted(r['layer'] for r in layers)==[0,3,24,27]
print('kinds',[r['kind'] for r in layers])
# The production full-attention label is normalized explicitly, not inferred by count.
for records in (layers,abl,roll):
    for r in records:
        if r['kind'] not in ('linear','full'):assert r['kind']=='attention';r['kind']='full'
proxy={str(T):{mode:weighted(layers,lambda r:next(t for t in r['timing'] if t['T']==T)['median_ms'][mode])
    for mode in ('bf16','w8a8')} for T in (1,2,4)}
ablation={mode:weighted(abl,lambda r:r['timing']['median_ms'][mode]) for mode in abl[0]['timing']['median_ms']}
restore={mode:{str(c):weighted(roll,lambda r:r['modes'][mode]['cuts'][str(c)]['median_ms']) for c in range(5)} for mode in ('bf16','w8a8')}
storage=weighted(roll,lambda r:r['modes']['w8a8']['prefix_storage_bytes'])
head=[];eq=rows=0;maxerr=0
for d in (0,1):
    j=load('head',d)
    for r in j['records']:
        n=sum(x['rows'] for x in r['output_errors']);k=sum(x['argmax_equal_rows'] for x in r['output_errors'])
        rows+=n;eq+=k;maxerr=max(maxerr,max(x['logit_error']['relative_l2'] for x in r['output_errors']))
        a,b=(r['median_ms'][k] for k in ('production','a8w8'))
        head.append(dict(device=d,T=r['T'],production_ms=a,a8w8_ms=b,saved_ms=a-b,latency_reduction=1-b/a,argmax_equal_rows=k,rows=n))
maxdiff={}
prefix_checks=continuations=skipped=0
for kind in ('linear','full'):
    vals=[s for r in layers if r['kind']==kind for v in r['validation'].values() for s in v['samples']]
    maxdiff[kind]=max(s['batch_vs_serial_output']['relative_l2'] for s in vals)
    for s in vals:
        for cut in s['prefix_cuts']:
            assert cut['state_exact'];prefix_checks+=1
            if cut['continuation_exact'] is None:skipped+=1
            else:assert cut['continuation_exact'];continuations+=1
head4=statistics.mean(x['a8w8_ms'] for x in head if x['T']==4)
target_proxy=proxy['4']['w8a8']+head4
budgets=[]
for p in (.5,.7,.8,.9,.95,1.):
    expected=sum(p**i for i in range(4));cap=33.01*expected/1.2
    budgets.append(dict(assumed_acceptance=p,assumed_expected_tokens=expected,cycle_budget_ms_for_20percent_speed=cap,
        four_layer_proxy_plus_new_head_ms=target_proxy,remaining_ms_before_drafter_IPC_restore=cap-target_proxy))
summary=dict(status='online_components_only_head_candidate_positive_verifier_cycle_unproven',exclusive=False,
    weighted_48layer_proxy_ms=proxy,weighted_v4_journal_ablation_ms=ablation,
    weighted_cached_restore_proxy_ms=restore,estimated_v4_prefix_storage_bytes=storage,
    head_components=head,head_argmax_equal_rows=eq,head_rows=rows,max_head_logit_relative_l2=maxerr,
    batch_vs_serial_max_output_relative_l2=maxdiff,prefix_state_checks=prefix_checks,
    continuation_checks=continuations,boundary_continuations_skipped=skipped,
    conditional_mtp_budget=budgets,packed_conv_adopted=False,head_deployed=False,
    full_model_quality_passed=False,new_http_tpot_measured=False,
    caveats=['Only L00/L03/L24/L27 with synthetic hidden and zero 8k KV; not 48 actual layers or full cycle.',
      'Event medians include graph execution gaps; nonexclusive parked production shares device.',
      'Restore uses repeatedly hot state; excludes CPU decision, RPC and full-model synchronization.',
      'Conditional acceptance model and historical r157 33.01ms are planning assumptions, not measurements.',
      'Head numerics changed; synthetic argmax is not task quality; complete frozen evaluation required.'])
(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
