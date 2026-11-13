"""Publish a stopped component experiment without implying later stages ran."""
import hashlib,json,shutil,subprocess
from pathlib import Path

P=Path('/models/fnx-diagnostics/r159-bf16-down-validation-20261114')
REPO=Path('/root/qwen-ascend')
EXP=REPO/'experiments/r159-bf16-down-validation'
BENCH=REPO/'benchmarks/r159-bf16-down-validation'
final=json.loads((P/'final-validation.json').read_text())
assert final['within_hour'] and final['private_env_removed'] and final['temporary_controls_removed']
assert final['production_files_unchanged']==45
assert not EXP.exists() and not BENCH.exists()
selection=json.loads((P/'component-selection.json').read_text())
assert not selection['passed'] and selection['selected_tile'] is None
assert not (P/'layers-device0.json').exists() and not (P/'baseline/summary.json').exists() and not (P/'combined/summary.json').exists()
components=[json.loads((P/f'component-device{d}.json').read_text()) for d in (0,1)]
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
metrics=[]
for r in components:
    med=r['median_us'];samples=r['samples_us']
    metrics.append(dict(device=r['device'],layer=r['layer'],median_us=med,
        rounded256_slowdown_percent=100*(med['rounded256']/med['production']-1),
        extra_us_vs_old=med['rounded256']-med['old'],
        rounded256_slower_rounds=sum(a>b for a,b in zip(samples['rounded256'],samples['production'])),
        rounds=len(samples['production']),
        max_diagonal_relative_l2=max(x['diagonal']['256']['relative_l2'] for x in r['records']),
        max_merge_relative_l2=max(x['merge']['rounded256']['relative_l2'] for x in r['records']),
        old_max_merge_relative_l2=max(x['merge']['old']['relative_l2'] for x in r['records']),
        minimum_diagonal_exact_fraction=min(x['diagonal']['256']['exact_fraction'] for x in r['records']),
        experts_covered=r['experts_covered'],samples=len(r['records']),
        fresh_input_graph_verified=r['fresh_input_graph_verified'],peak_allocated_bytes=r['peak_allocated_bytes']))
analysis=dict(status='rejected_at_component_performance_gate',candidate_deployed=False,
    component_metrics=metrics,selection=selection,full_layer_test_run=False,full_model_test_run=False,
    new_tpot_measured=False,old18k_regression_retested=False,full_quality_acceptance=False,
    production_revision='r152',final_validation=final)
(P/'summary.json').write_text(json.dumps(analysis,indent=2)+'\n')
rows=[[f'{r["device"]} / L{r["layer"]:02d}']+[f'{r["median_us"][k]:.3f}' for k in ('production','old','rounded128','rounded256')] for r in metrics]
report=['# r159：BF16 舍入修正版未通过组件性能门槛','',
    '2026-11-14，2×Ascend 910B4-1，同一固定 PCI 对，当前物理3/5、runtime逻辑0/1。起点提交7871863。','',
    '**修正版改善了组件误差，但完整 down+epilogue 慢于生产路径。按预先冻结的门槛提前停止；未进入整层、旧18k整模型回归或HTTP TPOT测试，正式保留r152。**','',
    '## 实测','',table(['逻辑卡 / 层','生产 µs','旧直接down µs','修正tile128 µs','修正tile256 µs'],rows),'']
for m in metrics:
    report += [f'逻辑卡{m["device"]}：最佳修正版tile256比生产慢{m["rounded256_slowdown_percent"]:.2f}%，比旧直接down增加{m["extra_us_vs_old"]:.3f}µs。16轮中{m["rounded256_slower_rounds"]}轮慢于同轮生产路径。','']
report += ['这是单个真实L00/L24权重的独占设备 down+epilogue 结果，不能直接推断整模型TPOT。生产路径含现有W4A16计算及原epilogue；旧路径是r156直接INT4 down加对角epilogue；修正版使用r158同一二进制，先恢复INT4×scale的BF16权重舍入，再使用同一对角epilogue。','',
    '每卡53组输入、四档幅度，覆盖全部512专家及重复专家511；时序使用64 banks覆盖整组权重，每张NPUGraph包含64次调用，每轮replay两次（共128次），16轮正反交替测序。另行改变激活、专家ID、路由权重、shared和gate，验证fresh graph/eager逐位相同。时序不与生产模型共享设备；用NPU Event区间计时，包含图内执行和间隙，不等于纯kernel求和或HTTP耗时。','',
    '## 数值与停止条件','',
    table(['逻辑卡','旧merge最大relative L2','修正merge最大relative L2','修正diagonal最大relative L2'],[[m['device'],f'{m["old_max_merge_relative_l2"]:.9f}',f'{m["max_merge_relative_l2"]:.9f}',f'{m["max_diagonal_relative_l2"]:.9f}'] for m in metrics]),'',
    '每卡峰值张量分配515901440字节（约492MiB），allocator上限为1/32；这是allocated，不是HBM或设备峰值带宽计数。修正版仍有少量BF16差异，不宣称与生产逐位一致。','',
    '预先冻结的性能门槛：tile128/256中至少一档在两卡均比生产down+epilogue快2%，才选型进入真实层。两档均未满足，因此selected_tile=null。没有放宽门槛，也没有用原r157的27.96/28.54ms冒充修正版TPOT。完整质量验收仍然缺失，不能宣称旧18k回归已修复。','',
    '## 执行与恢复','',
    '窗口自2026-11-14 00:58:44 CST开始，上限01:58:44；40分钟绝对timer及43分钟独立guardian，重试没有重置截止。首次systemd append日志打开失败发生在停服前，改为进程内重定向；首次单权重探针又因继承审计钩子未登记层号，在计时前退出并触发独立恢复。失败源码、哈希和日志保留。补登记只改探针元数据，未改模型、kernel、评分公式或阈值；修正后源文件在实际计时前重新冻结。','',
    '第一次恢复已验证健康后才在原窗口续跑。根据4.5分钟左右的实际模型启动成本，将可选4段NLL与10个自然切片延后，以优先保留旧五切片和HTTP；这项预算变更也发生在任何质量或性能样本产生前。但组件门槛随后失败，这些后续阶段均未执行。','',
    f'最终恢复完成于窗口开始后{final["restore_elapsed_s"]:.1f}秒，早于60分钟上限。45个生产文件及p275保持，原容器保留；9002 r152、manager、单watchdog和fresh8k原回答均验证通过。本轮私有ENV、lease、drop-in、timer和实验进程清理，正常boot-restore保留。','',
    '## 后续依据','',
    '这次排除了“直接加BF16舍入即可保留旧down收益”的实现，尚未排除其他算子布局或整模型方案。新增舍入引入的Mul/Cast与调度差异需要重新设计；没有新的PMU数据，不能把这约20µs增量全归因于HBM、L2或某条指令。后续先证明新的组件实现有净收益，再进入整层和整模型。','',
    'r157的W8A8单独版本29.38/30.07ms与旧组合27.96/28.54ms仍仅为历史数据；前者尚未通过完整质量验收，后者存在旧18k回归，两者都未被本轮自动发布。','',
    '证据：benchmarks/r159-bf16-down-validation；执行源：experiments/r159-bf16-down-validation。公开包没有模型权重、完整ENV、原始任务/文本/token IDs或凭据。','']
(P/'report.md').write_text('\n'.join(report))
EXP.mkdir(parents=True);BENCH.mkdir(parents=True)
manifest=json.loads((P/'harness-manifest.json').read_text())
source_names=set(f for f in manifest if f.endswith(('.py','.sh')))
source_names.update(('finalize_completed_run.py','publish_results.py'))
for name in sorted(source_names):shutil.copyfile(P/name,EXP/name)
shutil.copyfile(P/'README.md',EXP/'planned-scope.md')
(EXP/'README.md').write_text('''# r159 BF16 down 实测：组件门槛未通过

实际结果见../../docs/r159-bf16-down-validation.md与相邻benchmarks目录。生产继续r152。

本机本轮执行脚本，路径固定，不是新环境一键启动器。首次start_window.py的systemd append日志打开失败；实际通过launch-retry.json/measurement-retry.json所记的进程内重定向启动。component_probe.py在初次审计登记错误后修正；原始版本与初次manifest在initial-component-attempt保留。

实际执行：prepare/preflight、单权重组件比较和select_component门槛、独立rescue、finalize_completed_run。controller_retry.sh计划的整层、NLL/prefill/HTTP后续阶段没有执行。planned-scope.md是起初计划，不代表已测覆盖。

组件门槛失败后未降低标准或进入整模型。不得在正式Engine占卡时运行独占探针，不得重置window-start沿用已经结束的授权。复现需新的明确时间窗口与独立目录，并保留原入口恢复保护。

依赖r157原私有runtime与r158固定二进制；prepare.py记录实际生成方式，candidate-kmajor-down.py与candidate-audit.py是本轮实际集成模块，runtime-code-manifest.json可核查。编译器/模型环境沿用前轮。完整模型和任务输入只在服务器，未纳入公开包。
''')
for name in ('kmajor_down.py','audit.py'):shutil.copyfile(P/'combined/fnx'/name,EXP/('candidate-'+name.replace('_','-')))
data_names=('component-device0.json','component-device1.json','component-selection.json','summary.json','report.md',
    'final-validation.json','production-http.json','production-before-hashes.json','runtime-code-manifest.json',
    'harness-manifest.json','harness-before-budget-revision.json','debug-policy.json','preflight.json','cpu-review.json',
    'window.json','window-start.txt','restore-complete.json','restored-health.json','initial-launch-error.txt',
    'instrumentation-fix.json','budget-revision.json','launch-retry.json','measurement-retry.json',
    'controller.log','component-device0.log','component-device1.log',
    'npu-after.txt','topology.txt','weight-source-manifest.json')
for name in data_names:
    if (P/name).exists():shutil.copyfile(P/name,BENCH/name)
shutil.copytree(P/'initial-component-attempt',BENCH/'initial-component-attempt')
docs=REPO/'docs';shutil.copyfile(P/'report.md',docs/'r159-bf16-down-validation.md')
tech=docs/'Qwen3.8-Flash-Next-2x910B4-技术文档.md'
old=tech.read_bytes()
tracked=subprocess.check_output(['git','show','HEAD:docs/Qwen3.8-Flash-Next-2x910B4-技术文档.md'],cwd=REPO)
assert old==tracked and b'<!-- r159-bf16-down-validation -->' not in old
prefix=('<!-- r159-bf16-down-validation -->\n'
    '## r159 BF16舍入修正未通过组件性能门槛（2026-11-14，正式保留r152）\n\n'
    '两卡真实L00/L24独占down+epilogue：生产60.448/60.746µs，旧直接down49.681/49.578µs；'
    'BF16舍入修正tile256为69.404/69.423µs，比生产慢14.8%/14.3%，tile128更慢。'
    '两卡共32轮修正版均慢于同轮生产路径。最终merge最大relative L2改善到0.000114/0.000149，'
    '但未通过预先冻结的至少2%提速门槛，因此未执行整层、旧18k回归或整模型TPOT，不能转记r157约18%收益。\n\n'
    f'窗口开始后{final["restore_elapsed_s"]:.1f}秒提前恢复r152；45个生产文件/p275、原容器保持，manager/单watchdog及fresh8k原回答验证通过，临时实验控制清理。'
    '启动器日志与探针审计登记的初次错误、修正和冻结记录保留。详见[r159实测记录](r159-bf16-down-validation.md)。\n'
    '<!-- /r159-bf16-down-validation -->\n\n').encode()
tech.write_bytes(prefix+old);assert tech.read_bytes()[len(prefix):]==old
(BENCH/'techdoc-block.md').write_bytes(prefix)
(BENCH/'technical-doc-update.json').write_text(json.dumps(dict(before_sha256=hashlib.sha256(old).hexdigest(),
    after_sha256=hashlib.sha256(tech.read_bytes()).hexdigest(),old_bytes=len(old),prefix_bytes=len(prefix),old_bytes_preserved_as_suffix=True),indent=2)+'\n')
for root in (EXP,BENCH):
    fs=sorted(f for f in root.rglob('*') if f.is_file() and f.name!='SHA256SUMS')
    (root/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(root))+'\n' for f in fs))
print('REJECTED_COMPONENT_EVIDENCE_PACKAGED',EXP,BENCH)
