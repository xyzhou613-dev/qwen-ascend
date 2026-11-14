"""Publish bounded component evidence and an unstarted full-model plan."""
import hashlib,json,shutil,subprocess
from pathlib import Path
P=Path(__file__).resolve().parent;R=Path('/root/qwen-ascend')
NEXT=Path('/models/fnx-diagnostics/r162-w8a8-head-quality-20261114')
S=json.loads((P/'summary.json').read_text());F=json.loads((P/'final-validation.json').read_text())
assert F['production_files_unchanged']==45 and F['original_pids_preserved'] and not F['service_interrupted']
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join('---' for _ in headers)+' |']+['| '+' | '.join(map(str,r))+' |' for r in rows])
def write(p,s):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(s)
report=['# r161：W8A8验证器、完整前缀回滚和A8W8头部组件实测','',
    '2026-11-14，起点b82e0fb；双910B4-1，物理3/5、逻辑0/1。正式r152全程在线，原两进程未重启。','',
    '**新head组件节省约1.1ms，保留为默认关闭的整模型候选；V4完整周期尚无收益证据，没有新HTTP TPOT或质量验收结论。**','',
    '## 测量范围','',
    '在线空闲阶段分别测试L00/L03与L24/L27，使用真实权重、合成hidden及8k零KV上下文；每次只构造两种版本的两个层对象，共享原始量化张量，不构造第二个Engine。allocator上限1/24，物理可用内存须大于2GiB；每轮检查r152健康及pipeline计数，发现生产请求推进即中止。','',
    '所有时序均为非独占NPU Event，含图内执行及间隙，不是纯kernel时间、HBM流量或HTTP TPOT。四层加权公式为36×两线性层均值＋12×两全注意力层均值；没有执行完整48层，也没有纳入drafter、head、IPC和同步。12轮正反交替，两次replay含16次层调用。','',
    '## 同轮验证器成本','',
    table(['T','BF16层加权代理 ms','W8A8层加权代理 ms'],[[t,f'{v["bf16"]:.3f}',f'{v["w8a8"]:.3f}'] for t,v in S['weighted_48layer_proxy_ms'].items()]),'',
    'T2/T4包含FP32递推状态和conv前缀日志、KV/索引受影响尾部备份；T1为正常decode。SmoothQuant固定alpha=.5，四个校准文件SHA保持，不重新拟合。本轮不得与旧r153的79.94ms直接相减：旧方法只记录S，输入及缓存条件不同。','',
    '## 回滚与数值边界','',
    f'补齐S、conv及全注意力KV/raw index/pooled index的前缀恢复，部分接受时恢复原pooled组并只重建保留前缀。T2/T4在4093、4096、4097和8k边界验证cut=0..T；{S["prefix_state_checks"]}个前缀状态检查、{S["continuation_checks"]}个后续单步检查通过，容量边界{S["boundary_continuations_skipped"]}个后续检查明确跳过。改变输入和position后的graph/eager状态与前缀一致。','',
    f'线性层批量与逐token输出及FP32 S在覆盖样本逐位相同；全注意力输出最大relative L2为{S["batch_vs_serial_max_output_relative_l2"]["full"]:.8f}，因此不能声称整个批量验证器与串行逐位等价。前缀测试针对批量路径自身的状态语义，不证明全模型推测解码或采样正确性。独立FP32逐步DeltaNet参考8组也通过。','',
    f'实际前缀buffer字节数按36/12层估计为{int(S["estimated_v4_prefix_storage_bytes"]):,}字节（{S["estimated_v4_prefix_storage_bytes"]/1024**2:.2f}MiB，两卡合计）。定时路径只复制O(T)状态或缓存行；完整8k CPU快照仅为数值验证fixture，排除在计时外。','',
    table(['保留cut','BF16热状态恢复代理 ms','W8A8热状态恢复代理 ms'],[[c,f'{S["weighted_cached_restore_proxy_ms"]["bf16"][str(c)]:.3f}',f'{S["weighted_cached_restore_proxy_ms"]["w8a8"][str(c)]:.3f}'] for c in range(5)]),'',
    '上述恢复重复使用同一组buffer，缓存有利；未含CPU接受决策、跨进程通知或整模型同步。cut4无设备恢复操作，记0；不代表整轮没有成本。','',
    '## 日志消融：拒绝单次packed conv拷贝','',
    table(['变体','BF16 V4代理 ms','W8A8 V4代理 ms'],[[name,f'{S["weighted_v4_journal_ablation_ms"]["bf16/"+name]:.3f}',f'{S["weighted_v4_journal_ablation_ms"]["w8a8/"+name]:.3f}'] for name in ('legacy','loop','packed')]),'',
    'legacy只保留S，是不完整回滚的成本对照；loop为完整状态日志；packed把T+1次conv切片拷贝改成unfold视图的一次copy。packed在覆盖样本与loop逐位相同，重跑前缀及动态graph通过，但同轮更慢，未采用。不能把不完整legacy作为可上线基线。','',
    '## 新head：有明确组件收益，尚未质量验收','',
    '当前生产head为四块DenseW8分组W8A16，K=2560、词表248320。新类按256输出列的小tile把现有分组权重还原到BF16，再量化为每输出通道int8；调用时对每token做一次动态A8，接四次npu_quant_matmul，BF16输出。没有新校准，没有原始BF16头部权重，也没有常驻完整BF16头部。','',
    table(['卡 / T','生产mixer+head+argmax ms','新路径 ms','节省 ms','合成argmax一致'],[[f'{x["device"]}/{x["T"]}',f'{x["production_ms"]:.3f}',f'{x["a8w8_ms"]:.3f}',f'{x["saved_ms"]:.3f}',f'{x["argmax_equal_rows"]}/{x["rows"]}'] for x in S['head_components']]),'',
    f'四档合成hidden，argmax合计{S["head_argmax_equal_rows"]}/{S["head_rows"]}行一致；logit relative L2最大{S["max_head_logit_relative_l2"]:.6f}。权重重建relative L2为0.00923472，新权重/scale共636692480字节；该测试峰值allocated为1388573696字节，不是HBM峰值流量。graph/eager及改变输入测试通过。组件耗时下降约61%，不能直接当作整模型TPOT改善。','',
    '## MTP预算判断','',
    '仅作设计预算：假设每轮最多4个有效token、各位置条件接受率p一致，则E=1+p+p²+p³；以历史r157 r152的33.01ms为参考，输出速度+20%的整轮预算为33.01×E/1.2。该接受率模型与历史基线不是本轮实测。','',
    table(['假设p','E','+20%整轮预算 ms','W8A8 V4代理+新head ms','草稿/IPC/恢复前余量 ms'],[[f'{x["assumed_acceptance"]:.2f}',f'{x["assumed_expected_tokens"]:.3f}',f'{x["cycle_budget_ms_for_20percent_speed"]:.3f}',f'{x["four_layer_proxy_plus_new_head_ms"]:.3f}',f'{x["remaining_ms_before_drafter_IPC_restore"]:.3f}'] for x in S['conditional_mtp_budget']]),'',
    'p=.8时代理成本已超预算；p=.9仍仅约1.6ms给全部额外成本。尚未测drafter、真实接受率、完整两进程IPC/同步及整轮吞吐，不能承诺MTP大幅提速，也不能凭四层代理最终否决专用验证器。当前优先评估W8A8+新head的M1整模型收益。','',
    '## 服务与下一阶段','',
    '45正式文件、p275、r152原进程PID141969/141988、原容器、manager、单watchdog及boot-restore均复核。生产pipeline计数整个探针阶段未推进，没有停服、正式重启、lease、临时drop-in或实验ENV；探针已退出。','',
    'r162准备了r152→W8A8→W8A8+head三路同配置fresh8k/262016、缓存及持续输出对照。head装配默认关闭；候选不含旧direct/LUT down或V4。短门槛为旧五切片各ΔNLL≤.05、head在两长度各节省≥.5ms、相对r152输出速度各提高≥14%，失败即恢复。通过才执行冻结的100352 NLL/28切片、1200题/五域2pp单侧95%保守配对界、10自然真prefill及旧五回归。1200标准答案（含200代码）及负格式判题自检全部通过，没有执行整模型。','',
    '基线已有196 NLL、10自然切片和999题，按哈希/配置一致且重新核验4域NLL哨兵及旧5切片的规则续补201题；候选完整重跑。旧检查点来源、SHA和中断尾行处理保留；不声称基线全量为本轮重新测得。','',
    '上一轮60分钟窗口已结束；r162计划需要新的最多6小时明确范围，目前只完成准备。5h35m绝对timer独立恢复、5h38m guardian备用，硬截止6h；先停止检查点客户端再取消判题，超时保存断点。控制器始终恢复r152，不自动发布候选。详见[r162计划](r162-w8a8-head-plan.md)。','',
    '公开证据只含探针源码、合成误差/时序及哈希；不含权重、校准向量、真实评测题/答案、token IDs、完整ENV或凭据。','']
write(P/'report.md','\n'.join(report))
E=R/'experiments/r161-quant-v4-head';B=R/'benchmarks/r161-quant-v4-head'
assert not E.exists() and not B.exists()
E.mkdir(parents=True);B.mkdir(parents=True)
for name in ('prepare.py','launch.py','launch_ablation.py','launch_head.py','launch_rollback.py','probe.py','verifier_state.py','small_delta.py','packed_journal.py','ablation.py','quant_head.py','head_probe.py','rollback_probe.py','finalize.py','analyze.py','publish.py'):
    shutil.copyfile(P/name,E/name)
for name in ('engine.py','smooth_w8a8.py'):
    (E/'runtime/fnx').mkdir(parents=True,exist_ok=True);shutil.copyfile(P/'runtime/fnx'/name,E/'runtime/fnx'/name)
write(E/'README.md','# r161 在线组件实验，未部署\n\n结果见../../docs/r161-quant-v4-head.md。脚本依赖现场r154私有校准与正式模型权重，路径固定；不包含这些私有输入。prepare.py从冻结r154 runtime构建副本，small_delta沿用r153。禁止在正式占卡时启动第二份完整Engine。\n\n所有结果exclusive:false；单层代理、head组件和热状态恢复均不能当作HTTP TPOT。packed conv已拒绝；QuantHead仅为需完整质量验收的候选。\n')
for d in (0,1):
    for kind in ('layers','ablation','head','rollback'):shutil.copyfile(P/f'{kind}-device{d}.json',B/f'{kind}-device{d}.json')
for name in ('summary.json','final-validation.json','runtime-manifest.json','production-before-hashes.json','calibration-reference.json','harness-manifest.json','ablation-harness-manifest.json','head-harness-manifest.json','rollback-harness-manifest.json','report.md'):shutil.copyfile(P/name,B/name)
shutil.copyfile(P/'report.md',R/'docs/r161-quant-v4-head.md')
# Publish the prepared harness, never private tasks, checkpoints, ENV or future authorization.
E2=R/'experiments/r162-w8a8-head-quality';B2=R/'benchmarks/r162-w8a8-head-preflight'
assert not E2.exists() and not B2.exists();E2.mkdir(parents=True);B2.mkdir(parents=True)
for f in NEXT.iterdir():
    if f.is_file() and f.suffix in ('.py','.sh'):shutil.copyfile(f,E2/f.name)
for name in ('model.py','audit.py','quant_head.py'):
    (E2/'candidate/fnx').mkdir(parents=True,exist_ok=True);shutil.copyfile(NEXT/'combined/fnx'/name,E2/'candidate/fnx'/name)
for name in ('README.md','acceptance-policy.json','run-policy.json'):shutil.copyfile(NEXT/name,E2/name)
for name in ('preflight.json','canonical-grader-summary.json','runtime-code-manifest.json','harness-manifest.json','baseline-reuse-provenance.json','production-before-hashes.json','natural-freeze.json','judge-image.json'):shutil.copyfile(NEXT/name,B2/name)
shutil.copyfile(NEXT/'README.md',R/'docs/r162-w8a8-head-plan.md')
tech=R/'docs/Qwen3.8-Flash-Next-2x910B4-技术文档.md';old=tech.read_bytes()
assert old==subprocess.check_output(['git','show','HEAD:docs/Qwen3.8-Flash-Next-2x910B4-技术文档.md'],cwd=R)
prefix=('<!-- r161-quant-v4-head -->\n## r161 验证器完整回滚与A8W8 head组件（2026-11-14）\n\n'
    '正式r152全程在线，45文件/p275及原两进程未改。真实L00/L03/L24/L27、合成hidden/8k零KV的非独占四层加权代理：W8A8 M1/M2/M4为28.823/62.444/92.199ms；已包含T2/T4完整前缀日志，未含drafter/head/IPC。'
    '256个前缀状态及240个后续单步检查通过；线性层覆盖样本与串行逐位相同，全注意力输出最大relative L2=.002794，不能称全验证器等价。V4前缀buffer估计550.77MiB两卡合计，热状态恢复代理4.61–5.77ms；packed conv拷贝更慢已拒绝。\n\n'
    '新A8W8头部从现有分组W8重建并重新按输出通道量化；mixer+head+argmax的M1两卡1.820/1.832→.713/.709ms，M4为1.894/1.905→.761/.759ms。合成argmax52/56一致，尚未完整质量验收；约61%组件降耗不等于HTTP TPOT提升，未部署。'
    'r162已准备三路整模型对照及冻结完整质量评测，1200标准答案判题自检通过；需要新停服范围，旧小时已结束。详见[r161实测](r161-quant-v4-head.md)及[r162计划](r162-w8a8-head-plan.md)。\n<!-- /r161-quant-v4-head -->\n\n').encode()
assert b'<!-- r161-quant-v4-head -->' not in old
# Compute the printed MiB from the audited storage size, including full-attention tails.
prefix=prefix.replace(b'550.77MiB',f'{S["estimated_v4_prefix_storage_bytes"]/1024**2:.2f}MiB'.encode())
tech.write_bytes(prefix+old);assert tech.read_bytes()[len(prefix):]==old
(B/'techdoc-block.md').write_bytes(prefix)
write(B/'technical-doc-update.json',json.dumps(dict(before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=hashlib.sha256(tech.read_bytes()).hexdigest(),old_bytes=len(old),prefix_bytes=len(prefix),old_bytes_preserved_as_suffix=True),indent=2)+'\n')
for root in (E,B,E2,B2):
    fs=sorted(f for f in root.rglob('*') if f.is_file() and f.name!='SHA256SUMS')
    write(root/'SHA256SUMS',''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(root))+'\n' for f in fs))
print('R161_COMPONENT_RESULT_AND_R162_PREPARATION_PACKAGED')
