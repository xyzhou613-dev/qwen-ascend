"""Recompute measured concurrency and rates from the emitted token timeline."""
import itertools,json,math,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
def analyze(path,label):
    if not path.exists():return None
    j=json.loads(path.read_text());groups={}
    for r in j['records']:
        b=r['batch'];events=r['timeline'];rows={i:[] for i in range(b)};steps={}
        for event in events:
            rows[event['row']].append(event['ns']);steps.setdefault(event['step'],[]).append(event)
        assert len(set(r['input_hashes']))==b and len(set(r['prefill_hidden_hashes']))==b
        for i,stamps in rows.items():
            assert len(stamps)==r['per_row'][i]['output_tokens']
            assert all(x<y for x,y in zip(stamps,stamps[1:]))
        intervals=[(v[0],v[-1]) for v in rows.values()]
        overlaps=sum(max(a[0],c[0])<min(a[1],c[1]) for a,c in itertools.combinations(intervals,2))
        union=(max(v[-1] for v in rows.values())-min(v[0] for v in rows.values()))/1e9
        count=sum(len(v)-1 for v in rows.values())
        assert math.isclose(count/union,r['total_decode_tps'],rel_tol=1e-12)
        for i,v in rows.items():
            assert math.isclose((len(v)-1)/((v[-1]-v[0])/1e9),r['per_row'][i]['decode_tps'],rel_tol=1e-12)
        simultaneous=sum(len(v)==b and len({e['ns'] for e in v})==1 for v in steps.values())
        row_tps=statistics.median(v['decode_tps'] for v in r['per_row'])
        gaps=sorted((y-x)/1e6 for x,y in zip(rows[0],rows[0][1:]))
        data=dict(rep=r['rep'],batch=b,input_tokens=r['input_tokens'],output_tokens=[len(v) for v in rows.values()],
            total_decode_tps=r['total_decode_tps'],per_row_tps=row_tps,tpot_ms=1000/row_tps,
            step_tpot_p50_ms=statistics.median(gaps),step_tpot_p95_ms=gaps[math.ceil(.95*len(gaps))-1],
            ttft_s=r['ttft_s'],end_to_end_output_tps=r['end_to_end_output_tps'],
            overlapping_row_pairs=overlaps,all_rows_emitted_steps=simultaneous,
            all_rows_distinct_input_hidden=True,output_texts_distinct=len({v['text'] for v in r['per_row']})==b,
            allocated_gib=r['allocated_gib'])
        if b==4:
            assert r['ops']==dict(moe_rows=4,gmm_groups=40,qsa_rows=4)
            assert overlaps==6 and simultaneous>=100,'NO_SUSTAINED_TRUE_BATCH4'
        groups.setdefault(r['requested_context'],[]).append(data)
    summary={}
    for ctx,records in groups.items():
        summary[str(ctx)]=dict(samples=len(records),input_tokens=records[0]['input_tokens'],
            total_decode_tps_median=statistics.median(r['total_decode_tps'] for r in records),
            total_decode_tps_range=[min(r['total_decode_tps'] for r in records),max(r['total_decode_tps'] for r in records)],
            per_row_tps_median=statistics.median(r['per_row_tps'] for r in records),
            tpot_ms_median=statistics.median(r['tpot_ms'] for r in records),
            ttft_s_median=statistics.median(r['ttft_s'] for r in records),
            end_to_end_output_tps_median=statistics.median(r['end_to_end_output_tps'] for r in records),
            records=records)
    return dict(label=label,status=j['status'],summary=summary)
old=analyze(P/'attempt-scatter/batch4-results.json','initial B4 scatter; chunk512')
nd4=analyze(P/'batch4-results.json','ND B4; chunk2048')
nd1=analyze(P/'batch1-results.json','ND B1; chunk2048')
comparisons={}
if nd4 and nd1:
    for ctx in nd4['summary'].keys()&nd1['summary'].keys():
        a=nd4['summary'][ctx];b=nd1['summary'][ctx]
        comparisons[ctx]=dict(total_throughput_gain_pct=100*(a['total_decode_tps_median']/b['total_decode_tps_median']-1),
            per_row_tpot_increase_pct=100*(a['tpot_ms_median']/b['tpot_ms_median']-1))
result=dict(initial_scatter=old,nd_batch4=nd4,nd_batch1=nd1,matched_path_comparisons=comparisons,
    caveats=['static full-model batch, not an HTTP continuous batching scheduler',
        'shared document prefix, distinct suffixes, EOS respected',
        'some B1-only fused operations disabled; unmodified BF16 head',
        'experimental numerical path; full quality evaluation not completed',
        'long context one group when present; no variance estimate',
        'initial/ND comparison also changes prefill chunk, not a pure writer A/B'])
(P/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
for name,v in (('OLD_B4',old),('ND_B4',nd4),('ND_B1',nd1)):
    if v:print(name,json.dumps({k:{n:s[n] for n in ('samples','input_tokens','total_decode_tps_median','per_row_tps_median','tpot_ms_median','ttft_s_median')} for k,s in v['summary'].items()}))
print('MATCHED',json.dumps(comparisons))
