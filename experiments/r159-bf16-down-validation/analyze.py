"""Summarize only measured pairs; a short screen can never authorize release."""
import hashlib,json,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
OLD=Path('/models/fnx-diagnostics/r157-combined-quality-20261113')
def load(path):return json.loads(path.read_text()) if path.exists() else None
def paired_file(name):
    base=load(P/f'baseline-{name}.json') or {}
    new=load(P/f'combined-{name}.json') or {}
    out={}
    for key in sorted(base.keys()&new.keys()):
        a,b=base[key],new[key]
        assert a['ids_sha256']==b['ids_sha256'] and a['scored']==b['scored'] and a['mode']==b['mode']
        out[key]=dict(baseline_nll=a['mean_nll'],candidate_nll=b['mean_nll'],delta=b['mean_nll']-a['mean_nll'],scored=a['scored'])
    return out
final=load(P/'final-validation.json')
assert final and final['within_hour'] and final['production_files_unchanged']==45
data=dict(status='debug_only_no_full_acceptance',candidate_deployed=False,
    production_revision='r152',quality_screen_scope='frozen regression and 2048-target screen; no 1200-task evaluation',
    component={str(d):load(P/f'component-device{d}.json') for d in (0,1)},
    component_selection=load(P/'component-selection.json'),layers=[],performance={},
    legacy_prefill=paired_file('prefill-quality'),natural_prefill=paired_file('natural'),nll_screen=paired_file('nll-screen'),
    final_validation=final)
for d in (0,1):
    value=load(P/f'layers-device{d}.json')
    if value:data['layers']+=value['records']
if len(data['layers'])==4:
    by_kind={kind:[r['median_ms'] for r in data['layers'] if r['kind']==kind] for kind in ('linear','full')}
    means={kind:{mode:statistics.mean(r[mode] for r in rows) for mode in ('w8a8','old','rounded')} for kind,rows in by_kind.items()}
    data['layer_means_ms']=means
    data['weighted_48_layer_saving_proxy_ms']=36*(means['linear']['w8a8']-means['linear']['rounded'])+12*(means['full']['w8a8']-means['full']['rounded'])
for phase in ('baseline','combined'):
    data['performance'][phase]=load(P/phase/'summary.json')
if all(data['performance'].values()):
    data['performance_comparison']={}
    max_difference=0
    for phase in ('baseline','combined'):
        for name in ('8k.json','256k.json'):
            rows=load(P/phase/name)
            for row in rows:
                stats=row['stats']
                client=(row['end_to_end_s']-row['ttft_s'])*1000/(stats['tokens']-1)
                max_difference=max(max_difference,abs(client-stats['tpot_ms']))
    for key in ('median_8k_tpot_ms','median_256k_tpot_ms'):
        a=data['performance']['baseline'][key];b=data['performance']['combined'][key]
        data['performance_comparison'][key]=dict(saved_ms=a-b,latency_reduction_percent=100*(1-b/a),output_speed_gain_percent=100*(a/b-1))
    data['max_client_stats_tpot_abs_delta_ms']=max_difference
    for name in ('8k.json','256k.json'):
        a,b=load(P/'baseline'/name),load(P/'combined'/name)
        assert [r['payload_sha256'] for r in a]==[r['payload_sha256'] for r in b]
        old=load(OLD/'rollback'/name)
        assert [r['answer'] for r in a]==[r['answer'] for r in old], 'baseline answer reproduction failed'
    oldprefill=load(OLD/'combined-prefill-quality.json')
    for name,row in data['legacy_prefill'].items():
        row['old_combined_delta']=oldprefill[name]['mean_nll']-row['baseline_nll']
    data['legacy_prefill_all_within_limit']=len(data['legacy_prefill'])==5 and all(r['delta']<=.05 for r in data['legacy_prefill'].values())
    data['natural_prefill_all_within_limit']=len(data['natural_prefill'])==10 and all(r['delta']<=.05 for r in data['natural_prefill'].values())
(P/'analysis.json').write_text(json.dumps(data,indent=2)+'\n')
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
report=['# r159：BF16 权重舍入修正的组件与整模型实测','',
    '**正式保留 r152；本轮短筛查不能替代完整质量验收，候选没有发布。**','',
    '原 r158 同一二进制用于本轮私有 wrapper。down 先舍入 BF16 权重，随后沿用原对角 epilogue；没有将 r157 的 TPOT 转记为新结果。','',
    '## Down + epilogue','']
rows=[]
for d,r in data['component'].items():
    if r:rows.append([d]+[f'{r["median_us"][k]:.3f}' for k in ('production','old','rounded128','rounded256')])
report+=[table(['逻辑卡','生产 µs','旧 down µs','BF16 tile128 µs','BF16 tile256 µs'],rows),'',
    '53 组输入覆盖512专家与重复ID、四档幅度。64 banks 图时序覆盖全部权重，16轮正反交替测量；fresh graph/eager检查改变激活、专家、路由权重、shared和gate。这里测完整down+epilogue，不能用其下降比例直接推算整模型。','',
    '组件选型/门槛：`'+json.dumps(data['component_selection'],ensure_ascii=False)+'`。','',
    '## 四个真实层','']
report+=[table(['层','类型','W8A8原down ms','旧down ms','BF16修正 ms'],[[r['layer'],r['kind']]+[f'{r["median_ms"][k]:.6f}' for k in ('w8a8','old','rounded')] for r in data['layers']]),'',
    '输出误差与conv/FP32递推或KV/index状态检查见analysis.json；加权层代理不是HTTP实测。','',
    '## 整模型','']
if all(data['performance'].values()):
    report += [table(['场景 / TPOT ms','同轮r152','W8A8+BF16修正'],[[name]+[f'{data["performance"][phase][key]:.2f}' for phase in ('baseline','combined')] for name,key in (('fresh8k / 128输出','median_8k_tpot_ms'),('fresh262016 / 128输出','median_256k_tpot_ms'),('长文缓存 / 128输出，单次','cached_256k_tpot_ms'),('持续输出上限1024，单次','sustained1024_tpot_ms'))]),'']
    for name,key in (('8k','median_8k_tpot_ms'),('长文','median_256k_tpot_ms')):
        c=data['performance_comparison'][key]
        report += [f'{name}耗时下降{c["latency_reduction_percent"]:.2f}%，输出速度提高{c["output_speed_gain_percent"]:.2f}%。','']
    report += ['fresh各三次中位数，输入SHA一致，实际输入8064/262016、n_prefill8060/262012，均输出128。持续请求实际输出分别为'+ '/'.join(str(data['performance'][phase]['sustained_actual_tokens']) for phase in ('baseline','combined'))+'，未假称固定1024输出。TTFT和启动时间单独记录于各summary/startup.json。','']
else:report += ['整模型对照未完成，不给出修正版TPOT收益。组件门槛失败、错误或截止情况见controller.log及component-selection.json。','']
report+=['## 固定质量切片','',table(['旧真prefill切片','r152 NLL','修正候选 Δ','旧组合 Δ'],[[name,f'{r["baseline_nll"]:.6f}',f'{r["delta"]:+.6f}',f'{r["old_combined_delta"]:+.6f}' if 'old_combined_delta' in r else '—'] for name,r in data['legacy_prefill'].items()]),'',
    table(['r157自然文本回归','r152 NLL','修正候选 Δ'],[[name,f'{r["baseline_nll"]:.6f}',f'{r["delta"]:+.6f}'] for name,r in data['natural_prefill'].items()]),'',
    '旧五切片每片上限+0.05nat；r152需在5e-5内复现r157旧值。另有4段2048目标NLL短筛查。r157的1200任务及100352目标没有对新候选重跑，域级置信界不能据此视为通过。旧18k已参与数值修正开发，r157自然文本也已测过，本轮不宣称它们是新的未见holdout。原冻结质量门槛保持不变。','',
    '## 恢复','',f'独立恢复完成于开始后{final["restore_elapsed_s"]:.1f}秒，在60分钟内。45个生产文件、p275、原容器保持；9002/manager/单watchdog恢复，临时控制与私有ENV清理，fresh8k回答与基线一致。正常boot-restore保留。','',
    '本轮未采集新的PMU账本，未从组件时序推断HBM饱和度、Cube/Vector峰值效率或IPC占比。','']
(P/'report.md').write_text('\n'.join(report))
print('ANALYZED_DEBUG_ONLY_NOT_RELEASED',data.get('performance_comparison'),flush=True)
