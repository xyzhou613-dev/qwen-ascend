"""Package completed/stopped r162 measurements, preserving private evaluation data."""
import hashlib,json,shutil,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent;R=Path('/root/qwen-ascend')
def read(n,default=None):
    f=P/n;return json.loads(f.read_text()) if f.exists() else default
def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def table(h,rows):return '\n'.join(['| '+' | '.join(h)+' |','| '+' | '.join('---' for _ in h)+' |']+['| '+' | '.join(map(str,x))+' |' for x in rows])
F=read('final-validation.json');assert F and F['within_six_hour_cap'] and F['temporary_controls_removed'] and F['production_files_unchanged']==45
assert not (P/'lease.active').exists() and not (P/'original-private.json').exists()
aux_final=read('attribution-final-validation.json')
if (P/'attribution-harness-manifest.json').exists():assert aux_final and aux_final['own_unit_removed'] and aux_final['helper_processes_absent']
A=read('paired-analysis.json');assert A
policy_proof=read('policy-integrity.json');assert policy_proof and policy_proof['same_as_frozen_r157']
assert sha(P/'acceptance-policy.json')==policy_proof['acceptance_policy_sha256']==read('run-policy.json')['full_quality_policy_sha256']
assert sha(P/'run-policy.json')==policy_proof['run_policy_sha256']
calroot=Path('/models/fnx-diagnostics/r154-w8a8-gemv-20261113')
cal=json.loads((calroot/'calibration-manifest.json').read_text())
assert len(cal['layers'])==48 and all(sha(calroot/'private-calibration'/x['file'])==x['sha256'] for x in cal['layers'])
assert all(sha(P/n)==v['sha256'] for n,v in read('input-manifest.json').items())
quality_inputs=read('quality-input-integrity.json');assert quality_inputs['matches_frozen_r157_analysis']
assert all(sha(P/n)==h for n,h in quality_inputs['files'].items())
gate=read('short-gate.json',{})
performance={s:read(s+'/summary.json',{}) for s in ('rollback','smooth','combined')}
full_http_protocol=False;long_recall=False
if all(performance.values()):
    v=A['measurement_validation']['fresh_execution']
    full_http_protocol=all(v[n][phase]['completed']==3 and v[n][phase]['all_prefilled'] and v[n][phase]['all_output_128'] for n in ('8k.json','256k.json') for phase in ('rollback','smooth','combined'))
    assert full_http_protocol,'INVALID_FRESH_MEASUREMENT_PROTOCOL'
    long_recall=all(s['long_recall_hits']==[3,3,3] for s in performance.values()) and all(read(phase+'/cached256k.json')['recall_hits']==3 for phase in performance)
accepted=bool(A['passed'] and gate.get('passed') and full_http_protocol and long_recall and F['formal_fresh8k_answer_identical'])
decision=('all_frozen_acceptance_gates_passed_r152_restored_pending_release_decision' if accepted else
          'rejected_short_gate_r152_restored' if gate and not gate['passed'] else
          'rejected_complete_quality_r152_restored' if A['complete'] and not A['passed'] else
          'incomplete_no_acceptance_r152_restored')
reason_cn=('性能及冻结质量门槛通过，已恢复r152，候选待正式上线决策' if accepted else
           '短筛查失败，提前恢复r152，未进入完整验收' if gate and not gate['passed'] else
           '完整配对已完成，未达到冻结质量验收标准，已恢复r152' if A['complete'] and not A['passed'] else
           '配对未完成，不予验收，已恢复r152')
summary=dict(status=decision,quality_status=A['status'],quality_complete=A['complete'],quality_passed=A['passed'],
    short_gate=gate or None,performance=performance,candidate_deployed=False,
    full_http_protocol_verified=full_http_protocol,long_fresh_and_cache_recall_passed=long_recall,
    acceptance_all_gates_passed=accepted,final_validation=F,
    attribution=read('attribution-complete.json'),attribution_final_validation=aux_final,
    baseline_reuse_provenance=read('baseline-reuse-provenance.json'),candidate_input_provenance=read('candidate-input-provenance.json'))
write(P/'summary.json',json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
perfrows=[]
for phase,label in (('rollback','r152 同轮基线'),('smooth','仅W8A8'),('combined','W8A8 + A8W8 head')):
    s=performance[phase]
    if s:perfrows.append([label,f'{s["median_8k_tpot_ms"]:.2f}',f'{s["median_256k_tpot_ms"]:.2f}',f'{s["median_8k_ttft_s"]:.3f}',f'{s["median_256k_ttft_s"]:.3f}',f'{s["cached_256k_tpot_ms"]:.2f}',f'{s["sustained1024_tpot_ms"]:.2f} ({s["sustained_actual_tokens"]} tokens)'])
report=['# r162：W8A8与A8W8头部的整模型分级实测','',
    f'2026-11-14；测试源码起点3ae9a1c，双910B4-1，物理3/5、逻辑0/1。**{reason_cn}。候选未部署。**','',
    '## 整模型性能','',
    table(['版本','8k TPOT ms','262016 TPOT ms','8k TTFT s','262016 TTFT s','缓存TPOT ms','持续输出TPOT ms（实际长度）'],perfrows),'',
    'fresh 8k与fresh262016各三次取中位数，输入逐文件SHA冻结；长文实际输入262016、输出上限128，不代表262144输入后再输出8192。缓存与持续输出各为单次，持续请求上限1024但实际生成长度随EOS变化，表中列明。TPOT为HTTP服务统计，另用SSE首文本/结束时间交叉核验。加载、装配及图捕获不计TTFT。','']
if all(performance.values()):
    for key,label in (('median_8k_tpot_ms','8k'),('median_256k_tpot_ms','262016')):
        b,s,c=(performance[k][key] for k in ('rollback','smooth','combined'))
        report += [f'{label}：新head相对W8A8节省{s-c:.3f}ms；组合相对同轮r152耗时降低{100*(1-c/b):.2f}%，输出速度提高{100*(b/c-1):.2f}%。','']
report += ['r161的1.83→.71ms仅为组件证据；本表才是本轮实际HTTP结果。不存在完整HTTP结果的阶段不补估算值。','',
    '## 装配、输入与短筛查','',
    '每阶段仅一个双进程Engine，私有19002；实际48层和96个SmoothW8A8投影由运行时对象审计，worker的四块QuantHead须为int8权重/float32 scale并进入捕获。候选保持原MoE down，不含旧direct/LUT down或V4。新head逐字节沿用r161组件实现，由原分组W8重建BF16后重新按输出通道量化，调用时per-token动态A8；默认关闭，不宣称数值等价。','',
    '48个校准文件SHA与原r154清单一致，alpha=.5固定；六份性能输入SHA与r157冻结清单一致。1200标准答案含200代码的隔离判题、负格式自检全通过，无新拟合或评测集调整。','',
    f'短筛查结果：{gate.get("passed","未完成")}。冻结要求：两长度各相对W8A8至少节省.5ms、相对r152输出速度至少提高14%；旧五个prefill切片各ΔNLL≤.05。失败立即进入恢复，不继续长跑。原始分项见short-gate.json。','',
    '## 冻结质量验收','',
    f'质量状态：{A["status"]}；完整性={A["complete"]}，通过={A["passed"]}。NLL配对{A["nll"]["pairs"]}/196，目标数{A["nll"]["scored"]}/100352；任务配对{A["tasks"]["paired"]}/1200；自然真prefill配对{A["fresh_natural_prefill"]["pairs"]}/10。未完成不能视作通过。','',
    '门槛原样沿用r157：全量平均ΔNLL≤.02、28预定切片各≤.05；1200任务按五域，2pp非劣界和单侧95%保守配对CP界，所有配对完整且无预定子集退化复核；10自然真prefill与旧五回归各ΔNLL≤.05，并验证计分parity。合成argmax、余弦或正常答题不替代这些标准。','']
if A['tasks']['complete']:
    base_correct=sum(d['baseline_correct'] for d in A['tasks']['domains'].values())
    candidate_correct=sum(d['candidate_correct'] for d in A['tasks']['domains'].values())
    report += [f'完整任务：r152 {base_correct}/1200，组合{candidate_correct}/1200，总体点差{100*(candidate_correct-base_correct)/1200:+.2f}pp。总体均值不替代各域门槛。','']
if A['nll']['pairs']:report += [f'当前NLL平均Δ={A["nll"].get("delta"):.8f}，完整={A["nll"]["complete"]}，门槛通过={A["nll"]["pass_limit"]}。','']
natural=A['fresh_natural_prefill']
if natural['pairs']:
    report += [f'自然真prefill {natural["pairs"]}/10、平均ΔNLL={natural["delta"]:+.8f}，门槛通过={natural["pass_limit"]}。','']
    bad=[x for x in natural['slices'] if x['delta']>A['policy']['natural_prefill_slice_nll_limit']]
    if bad:report += [table(['自然切片','基线NLL','候选NLL','ΔNLL'],[[x['name'],f'{x["baseline_nll"]:.8f}',f'{x["candidate_nll"]:.8f}',f'{x["delta"]:+.8f}'] for x in bad]),'']
domainrows=[]
for name,d in A['tasks']['domains'].items():
    lower=f'{100*d["conservative_one_sided_lower"]:+.3f}pp' if 'conservative_one_sided_lower' in d else '—'
    point=f'{100*d["delta"]:+.2f}pp' if 'delta' in d else '—'
    domainrows.append([name,f'{d["n"]}/{d["expected"]}',str(d.get('baseline_correct','—')),str(d.get('candidate_correct','—')),point,lower,str(d['pass_margin'])])
report += [table(['域','配对数','基线正确','候选正确','点差','保守下界','通过'],domainrows),'',
    '置信界未过说明这些冻结样本尚未证明2pp非劣，不必然表示已经证明真实准确率下降超过2pp；点估计持平或提高也不能替代预设置信界。','',
    '详细均值、切片、自然回归和任务置信界见paired-analysis.json。基线196 NLL、10自然及999任务来自r157旧检查点；要求45生产文件/p275与配置一致，四域NLL哨兵、旧五切片重测差≤5e-5，再补201题。来源SHA、复制SHA和中断尾行处理保留。候选重新测量；不声称整套基线在本轮重新执行。','',
    '## 时间窗口与恢复','',
    '用户本轮明确允许最多6小时。09:17:03起，14:52:03独立timer触发恢复、14:55:03备用guardian，15:17:03硬截止；没有重置绝对起点或使用前一轮已结束窗口。主控制器退出同样触发恢复。先终止检查点客户端，再清本run隔离判题容器，避免将取消的代码题计错；超时保留断点。','',
    f'实际恢复耗时距起点{F["restore_elapsed_s"]:.1f}秒，在6小时范围内。45正式文件/p275、原容器、r152 HTTP、manager、单watchdog及正常boot-restore均复核；fresh8k原回答一致，实验ENV/lease/drop-in/timer/判题容器/进程已清理。控制器不自动部署候选。','',
    '完整质量通过也仅说明本冻结集上的验收，不能排除预训练数据重叠，不覆盖原生tool-call协议或全面长文问答能力。公开文件去除生成答案/代码、解析结果、原始题文/tokenIDs及逐token argmax；保留输入SHA、计数、误差与时序。私有权重、校准向量和完整ENV不发布。','']
aux=read('attribution-complete.json')
if aux:
    report += ['## 超限切片的追加对照','',
        '完整1200任务落盘后，在原绝对窗口内暂停父控制器，沿用冻结W8A8 runtime补同一zh/65536切片，完成后恢复原控制器并触发原rescue。原timer/guardian和六小时截止不变，未改变模型算法、原质量检查点或验收阈值。此为后验归因，原候选失败保留。','',
        table(['同一切片路径','NLL','相对复用r152 ΔNLL'],[
            ['r152参考',f'{aux["baseline_nll"]:.8f}','0'],
            ['W8A8，原head',f'{aux["w8a8_only_nll"]:.8f}',f'{aux["w8a8_projection_delta"]:+.8f}'],
            ['W8A8＋新head',f'{aux["combined_nll"]:.8f}',f'{aux["combined_delta"]:+.8f}']]),'',
        f'同切片新head的增量ΔNLL为{aux["head_incremental_delta"]:+.8f}。组合重测逐token最大绝对差为{aux["repeated_combined_max_abs_token_delta"]}；null表示原Engine已进入退出流程，未补采重测。追加对照不替代完整质量验收。','']
elif (P/'attribution-status.json').exists():
    report += ['## 追加对照状态','',f'额外同切片对照未完成：{read("attribution-status.json")["stage"]}。原冻结质量判断和绝对截止保持。','']
write(P/'report.md','\n'.join(report))
B=R/'benchmarks/r162-w8a8-head-quality';E=R/'experiments/r162-w8a8-head-quality'
assert not B.exists();B.mkdir(parents=True)
for f in P.iterdir():
    if f.is_file() and f.suffix in ('.py','.sh'):shutil.copyfile(f,E/f.name)
write(E/'README.md',f'# r162 分级整模型验证结果\n\n{reason_cn}。候选未部署。实测见../../docs/r162-w8a8-head-quality.md；原冻结方案见../../docs/r162-w8a8-head-plan.md。\n\nstart.py和controller.sh使用本轮固定目录及一次性绝对窗口，不能重置截止或再次沿用此授权。私有模型权重、校准、完整质量题集及答案不在仓库。原冻结源码清单与派生分析来源保留；质量门槛未改变。\n')
def cleaned(obj):
    if isinstance(obj,list):return [cleaned(x) for x in obj]
    if isinstance(obj,dict):
        result={k:cleaned(v) for k,v in obj.items() if k not in ('answer','parsed','nll_tokens','argmax','messages','prompt','payload','input_ids','target_ids','generated_text','generated_code')}
        if 'answer' in obj:
            result['answer_sha256']=hashlib.sha256(obj['answer'].encode()).hexdigest()
            result['answer_characters']=len(obj['answer'])
        return result
    return obj
for phase in ('rollback','smooth','combined'):
    for name in ('summary.json','8k.json','256k.json','cached256k.json','sustained1024.json'):
        f=P/phase/name
        if f.exists():write(B/phase/name,json.dumps(cleaned(json.loads(f.read_text())),ensure_ascii=False,indent=2)+'\n')
    for name in (f'{phase}-runtime-audit.json',f'{phase}-startup.json',f'{phase}-scoring-parity.json',f'{phase}-prefill-quality.json',f'quality-{phase}-summary.json'):
        f=P/name
        if f.exists():write(B/name,json.dumps(cleaned(json.loads(f.read_text())),ensure_ascii=False,indent=2)+'\n')
    # Preserve independently recomputable correctness counts without answers/tests.
    taskfile=P/f'quality-{phase}.jsonl'
    if taskfile.exists():
        tasks=[]
        for line in taskfile.read_bytes().splitlines(keepends=True):
            if not line.endswith(b'\n'):continue
            row=json.loads(line)
            keep={k:row[k] for k in ('id','input_sha256','config_sha256','finish_reason','elapsed_s') if k in row}
            keep['correct']=bool(row['result']['correct']);tasks.append(keep)
        write(B/f'quality-{phase}-redacted.jsonl',''.join(json.dumps(x,separators=(',',':'))+'\n' for x in tasks))
    for kind in ('nll','natural'):
        f=P/f'{phase}-{kind}.jsonl'
        if f.exists():
            data=[cleaned(json.loads(x)) for x in f.read_bytes().splitlines(keepends=True) if x.endswith(b'\n')]
            write(B/f'{phase}-{kind}-redacted.jsonl',''.join(json.dumps(x,separators=(',',':'))+'\n' for x in data))
for name in ('summary.json','paired-analysis.json','short-gate.json','window.json','restore-complete.json','final-validation.json','baseline-reuse-repeat.json','baseline-reuse-provenance.json','candidate-input-provenance.json','input-manifest.json','policy-integrity.json','quality-input-integrity.json','analysis-lineage.json','production-before-hashes.json','runtime-code-manifest.json','harness-manifest.json','acceptance-policy.json','run-policy.json','canonical-grader-summary.json','natural-freeze.json','judge-image.json'):
    f=P/name
    if f.exists():shutil.copyfile(f,B/name)
for name in ('attribution-complete.json','attribution-runtime-audit.json','attribution-harness-manifest.json','attribution-window.json','attribution-status.json','attribution-pause.json','attribution-final-validation.json','attribution-unit.txt'):
    f=P/name
    if f.exists():shutil.copyfile(f,B/name)
for name in ('attribution-combined-repeat.json','attribution-smooth-result.json'):
    f=P/name
    if f.exists():write(B/name,json.dumps(cleaned(json.loads(f.read_text())),indent=2)+'\n')
shutil.copyfile(P/'report.md',B/'report.md');shutil.copyfile(P/'report.md',R/'docs/r162-w8a8-head-quality.md')
privatefiles=[f for f in P.iterdir() if f.is_file() and (f.suffix=='.jsonl' or f.name.endswith('.log') or f.name in ('attribution-combined-repeat.json','attribution-smooth-result.json'))]
write(B/'private-evidence-hashes.json',json.dumps({f.name:dict(bytes=f.stat().st_size,sha256=sha(f)) for f in privatefiles},indent=2)+'\n')
tech=R/'docs/Qwen3.8-Flash-Next-2x910B4-技术文档.md';old=tech.read_bytes()
assert old==subprocess.check_output(['git','show','HEAD:docs/Qwen3.8-Flash-Next-2x910B4-技术文档.md'],cwd=R)
perf_note=''
if all(performance.values()):
    b,s,c=(performance[k] for k in ('rollback','smooth','combined'))
    perf_note=(f'同轮r152→W8A8→W8A8+head，fresh8k TPOT {b["median_8k_tpot_ms"]:.2f}→{s["median_8k_tpot_ms"]:.2f}→{c["median_8k_tpot_ms"]:.2f}ms，262016为{b["median_256k_tpot_ms"]:.2f}→{s["median_256k_tpot_ms"]:.2f}→{c["median_256k_tpot_ms"]:.2f}ms；新head增量节省{s["median_8k_tpot_ms"]-c["median_8k_tpot_ms"]:.2f}/{s["median_256k_tpot_ms"]-c["median_256k_tpot_ms"]:.2f}ms，组合输出速度较r152提高{100*(b["median_8k_tpot_ms"]/c["median_8k_tpot_ms"]-1):.2f}%/{100*(b["median_256k_tpot_ms"]/c["median_256k_tpot_ms"]-1):.2f}%。')
prefix=('<!-- r162-w8a8-head-quality -->\n## r162 W8A8＋A8W8 head整模型分级验证（2026-11-14）\n\n'
    f'{reason_cn}；候选未部署。{perf_note}质量配对NLL {A["nll"]["pairs"]}/196、任务{A["tasks"]["paired"]}/1200、自然prefill {A["fresh_natural_prefill"]["pairs"]}/10；完整={A["complete"]}，通过={A["passed"]}。'
    '三组同配置HTTP及完整冻结质量门槛、基线检查点复用与实际恢复记录见[r162实测](r162-w8a8-head-quality.md)。组件加速不能替代整模型结果，未完成质量不能视作验收通过。\n\n'
    f'授权窗口09:17:03–15:17:03，14:52:03独立恢复，实际距起点{F["restore_elapsed_s"]:.1f}s恢复；45文件/p275不变，manager/单watchdog、fresh8k原回答及临时状态清理复验通过。\n<!-- /r162-w8a8-head-quality -->\n\n').encode()
assert b'<!-- r162-w8a8-head-quality -->' not in old
tech.write_bytes(prefix+old);assert tech.read_bytes()[len(prefix):]==old
(B/'techdoc-block.md').write_bytes(prefix)
write(B/'technical-doc-update.json',json.dumps(dict(before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=sha(tech),old_bytes=len(old),prefix_bytes=len(prefix),old_bytes_preserved_as_suffix=True),indent=2)+'\n')
for root in (B,E):
    files=sorted(f for f in root.rglob('*') if f.is_file() and f.name!='SHA256SUMS')
    write(root/'SHA256SUMS',''.join(sha(f)+'  '+str(f.relative_to(root))+'\n' for f in files))
print('R162_RESTORED_RESULTS_PACKAGED',decision)
