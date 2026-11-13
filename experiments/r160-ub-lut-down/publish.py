"""Archive an exact-to-r158 but slower UB-LUT implementation and restored service."""
import hashlib,json,shutil,subprocess
from pathlib import Path
P=Path('/models/fnx-diagnostics/r160-ub-lut-down-20261114');REPO=Path('/root/qwen-ascend')
EXP=REPO/'experiments/r160-ub-lut-down';BENCH=REPO/'benchmarks/r160-ub-lut-down'
final=json.loads((P/'final-validation.json').read_text())
assert final['original_hard_cap_preserved'] and final['temporary_controls_removed'] and final['production_files_unchanged']==45
assert not EXP.exists() and not BENCH.exists()
metrics=[]
for d in (0,1):
    r=json.loads((P/f'timing-device{d}.json').read_text());assert r['status']=='exclusive_component_complete'
    assert r['bitwise_equal_to_r158'] and r['experts_covered']==512
    m=r['median_us'];best=min(('lut128','lut256'),key=lambda k:m[k])
    metrics.append(dict(device=d,layer=r['layer'],median_us=m,best_lut=best,
        best_lut_latency_ratio=m[best]/m['production'],
        best_lut_slower_rounds=sum(a>b for a,b in zip(r['samples_us'][best],r['samples_us']['production'])),
        numerical_samples=r['samples'],experts_covered=r['experts_covered'],
        bitwise_equal_to_r158=True,fresh_graph_verified=r['fresh_graph_verified'],
        peak_allocated_bytes=r['peak_allocated_bytes']))
assert all(m['best_lut_latency_ratio']>1 for m in metrics)
summary=dict(status='exact_to_r158_but_rejected_on_component_cost',metrics=metrics,
    candidate_deployed=False,new_tpot_measured=False,full_layer_evaluated=False,
    old18k_regression_retested=False,full_quality_acceptance=False,final_validation=final)
(P/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
rows=[[f'{m["device"]} / L{m["layer"]:02d}']+[f'{m["median_us"][k]:.3f}' for k in ('production','old','rounded','lut128','lut256')] for m in metrics]
report=['# r160：片上 INT4→BF16 查表的数值与独占时序','',
    '2026-11-14，起点e44ed0a，原两张910B4-1，当前物理3/5；单权重组件，不构建第二份完整模型。','',
    '**新查表实现与r158逐位一致，但组件时序更慢，拒绝进入整层或整模型；正式继续r152，没有新TPOT结果。**','',
    '## 实现与数值','',
    '对每个64行K分组和输出列tile，先在UB内生成16个INT4取值×scale并舍入BF16；将原权重解包为索引后Gather这些BF16值，再沿用r158的FP32乘激活、分组归约树和原对角epilogue。没有常驻整模型LUT，没有重新打包权重，也没有把BF16舍入删掉。','',
    'tile128/256各有16×tile个BF16表项，分别占4/8KiB；对应FP32建表暂存8/16KiB。表和索引仅在UB存在，每个分组重新构造，不据此宣称全部UB占用只有这个表大小。','',
    '两卡分别用真实L00/L24 down；每次53组输入覆盖全部512专家、重复专家511与四档幅度。down输出及完整merge输出均与r158逐位相同。另改变激活、专家ID、路由权重、shared、gate，fresh graph/eager及对r158结果均逐位相同。在线数值阶段不采时序；独占计时前又完整复验一次。','',
    '这证明本探针覆盖下的r158数值等价，不是与r152全模型逐位等价，也没有证明旧18k回归已修复。两卡每个进程allocator上限1/32，峰值allocated均515901440字节（约492MiB）；不是物理HBM峰值或流量。','',
    '## 完整 down+epilogue 的独占时序','',
    table(['逻辑卡 / 层','生产 µs','旧直接down µs','r158舍入 µs','LUT128 µs','LUT256 µs'],rows),'']
for m in metrics:
    report += [f'卡{m["device"]}最佳查表版为{m["best_lut"]}，耗时是生产路径的{m["best_lut_latency_ratio"]:.3f}倍；16轮中{m["best_lut_slower_rounds"]}轮更慢。','']
report+=['五路同批比较，每张NPUGraph含64个专家bank调用，每轮replay两次，共128次down+epilogue；16轮正反交替测序。专家bank覆盖全部512权重，不反复仅命中10专家。NPU Event区间含图内执行及间隙，不等于纯kernel求和，也不是HTTP TPOT。两个正式Engine已暂停，计时不与正式模型共享设备。','',
    '本实现为保持BF16权重舍入增加了建表、整型转换、偏移计算与Gather。这些改动的净成本超过减少的乘法/转换；尚无PMU分项证据，不能把差值全归因于Gather吞吐、UB bank冲突、HBM或L2。小型片上表不必然比连续算术链快。','',
    '## 时间范围与恢复','',
    '用户在r159结束后要求继续；本轮短组件实测继续约束于已授权的原小时：00:58:44开始、01:38:44触发独立恢复、01:41:44 guardian备用、01:58:44硬截止。r160未重置起点或延长截止。新增短阶段从01:35:41左右开始，只做两卡组件，完成后立即由独立service恢复原入口。','',
    f'最终恢复距原小时起点{final["restore_elapsed_s_from_original_window"]:.1f}秒，距r160短阶段开始{final["micro_elapsed_s"]:.1f}秒，在原截止内。45个正式文件及p275未改，原容器保持；r152 HTTP、manager、单watchdog、fresh8k原回答一致及临时ENV/lease/drop-in/timer/实验进程清理均验证。正常boot-restore保留。','',
    '## 下一步依据','',
    '当前直接舍入版与本查表版都不作为整模型候选。要继续自研down，需改变数据通路和计算组织，先在同一真实shape证明净收益；不能仅因算式减少就进入全模型。生产ExpW4.apply已调用W4A16 Cube，简单将现有接口改成三维batch也受产品约束，项目r50已有相关记录。','',
    '针对更大TPOT收益，后续可优先将现有W8A8纳入r153的V4验证器实测，并补齐drafter、head、IPC和回滚成本；不把M1的节省从旧V4代理直接相减。完整周期与接受率尚未测，不承诺MTP收益。W8A8单独版本仍需完整质量验收。','',
    '接口核实参考：[Cast的A2类型组合](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0073.html)、[Gather](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0092.html)、[CreateVecIndex](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0090.html)、[WeightQuantBatchMatmulV2产品约束](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/API/aolapi/context/aclnnWeightQuantBatchMatmulV2.md)。是否能跑仍以本机编译与设备测试为准。','',
    '公开证据不含权重、完整进程ENV、原始任务/文本/token IDs或凭据。','']
(P/'report.md').write_text('\n'.join(report))
EXP.mkdir(parents=True);BENCH.mkdir(parents=True)
source=('lut_down.cpp','libfnx_w4_lut.so','build.sh','probe.py','launch.py','micro_controller.sh','start_micro.py',
    'rescue.sh','safety.py','snapshot.py','stop_owned.py','idle_guard.py','formal_smoke.py','http_bench.py','finalize.py','publish.py')
for name in source:shutil.copyfile(P/name,EXP/name)
(EXP/'README.md').write_text('''# r160 UB LUT down（组件更慢，未部署）

实际结果见../../docs/r160-ub-lut-down.md。代码在本机CANN9.1编译，53样本×512专家数值与r158逐位相同，动态graph通过；独占down+epilogue比生产更慢，未测真实层/全模型。

本机固定路径脚本，不是新环境一键启动器。probe.py默认仅做在线数值验证，并在发现生产请求进展时退出，不构造Engine；--timing只允许独占状态及有效lease/截止。单个真实down权重、1/32 allocator，禁止在正式模型占卡时假称独占结果。

start_micro.py只沿用当时明确授权的r159原绝对截止，不能重置或挪用已结束的窗口；后续复现需要新的有效范围及独立目录。原入口、manager、watchdog由独立rescue恢复；绝对timer和独立guardian保留至恢复完成，随后finalize清理并核验。

恢复后formal_smoke只执行http_bench.py的SSE前缀函数，未执行其中整套候选性能请求。不要把恢复烟测称为新候选TPOT。
''')
data=('numeric-device0.json','numeric-device1.json','timing-device0.json','timing-device1.json',
    'numeric-device0.log','numeric-device1.log','timing-device0.log','timing-device1.log','controller.log','build.log',
    'window.json','window-start.txt','source-manifest.json','production-before-hashes.json','production-http.json',
    'restore-complete.json','restored-health.json','final-validation.json','summary.json','report.md','topology.txt','npu-after.txt',
    'weight-source-manifest.json')
for name in data:
    if (P/name).exists():shutil.copyfile(P/name,BENCH/name)
docs=REPO/'docs';shutil.copyfile(P/'report.md',docs/'r160-ub-lut-down.md')
tech=docs/'Qwen3.8-Flash-Next-2x910B4-技术文档.md';old=tech.read_bytes()
assert old==subprocess.check_output(['git','show','HEAD:docs/Qwen3.8-Flash-Next-2x910B4-技术文档.md'],cwd=REPO)
assert b'<!-- r160-ub-lut-down -->' not in old
prefix=('<!-- r160-ub-lut-down -->\n## r160 片上INT4→BF16查表：数值相同，组件更慢（2026-11-14）\n\n'
    '已实现按分组/列tile在UB生成16项BF16表后Gather的down，沿用r158归约树和原epilogue；两卡真实L00/L24、各53样本覆盖512专家，down及merge与r158逐位相同，动态graph通过。'
    '独占完整down+epilogue：生产60.824/60.910µs，r158舍入69.275/69.355µs，查表tile128为173.447/173.834µs，tile256为263.558/264.114µs。最佳查表版仍约2.85倍耗时，拒绝进入整层/整模型，没有新TPOT或旧18k结论。\n\n'
    f'沿用已授权原小时的绝对截止，新增阶段后{final["micro_elapsed_s"]:.1f}秒恢复r152；45正式文件/p275保持，manager/单watchdog及fresh8k回答一致通过，临时控制清理。'
    '后续不再据减少算式承诺提速，应先验证新的计算组织；大幅TPOT方向可继续r153 V4+量化与完整周期预算。详见[r160实测](r160-ub-lut-down.md)。\n<!-- /r160-ub-lut-down -->\n\n').encode()
tech.write_bytes(prefix+old);assert tech.read_bytes()[len(prefix):]==old
(BENCH/'techdoc-block.md').write_bytes(prefix)
(BENCH/'technical-doc-update.json').write_text(json.dumps(dict(before_sha256=hashlib.sha256(old).hexdigest(),
    after_sha256=hashlib.sha256(tech.read_bytes()).hexdigest(),old_bytes=len(old),prefix_bytes=len(prefix),old_bytes_preserved_as_suffix=True),indent=2)+'\n')
for root in (EXP,BENCH):
    fs=sorted(f for f in root.rglob('*') if f.is_file() and f.name!='SHA256SUMS')
    (root/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(root))+'\n' for f in fs))
print('R160_NEGATIVE_COMPONENT_RESULT_PACKAGED')
