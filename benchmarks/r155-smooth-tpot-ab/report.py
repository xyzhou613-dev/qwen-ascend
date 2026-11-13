import json,statistics
from pathlib import Path
P=Path(__file__).resolve().parent;s=json.load(open(P/'summary.json'));rows=[]
names={'median_8k_tpot_ms':'fresh 8k / 128输出','median_256k_tpot_ms':'fresh 262016 / 128输出','cached_256k_tpot_ms':'长文缓存命中 / 128输出','sustained1024_tpot_ms':'fresh 8k / 最多1024输出'}
for key,label in names.items():
 r=s['metrics'][key]
 rows.append(f"| {label} | {r['baseline']:.2f} | {r['candidate']:.2f} | {r['rollback']:.2f} | {r['speedup_vs_baseline']*100:+.2f}% / {r['speedup_vs_rollback']*100:+.2f}% |")
tt=[]
for key,label in [('median_8k_ttft_s','fresh 8k'),('median_256k_ttft_s','fresh 262016')]:
 a,b,c=[s['phases'][p][key] for p in ('baseline','candidate','rollback')]
 tt.append(f'| {label} | {a:.3f} | {b:.3f} | {c:.3f} |')
q=json.load(open(P/'quality-reference.json'))
audit=json.load(open(P/'candidate-projection-audit.json'))
report=f'''# r155：SmoothQuant W8A8 整模型 TPOT A/B/A 实测

2026-11-13，192.168.18.206，2×Ascend910B4-1，物理3/5、容器0/1，24+24两进程，窗口262144。CANN9.1.0、驱动26.1.1、torch2.10.0、torch_npu2.10.0.post4。起点f9284fa / 正式r152。

本轮验证r154的校准后W8A8真实整模型收益，不把单层代理当TPOT。本轮没有接入直接MoE down GEMV：其权重布局切换开销尚未解决。实验结束恢复正式r152，候选不发布为生产版本。

## TPOT（ms）

| 场景 | 基线r152 | SmoothQuant候选 | 回切r152 | 输出速度提升：对基线 / 回切 |
|---|---:|---:|---:|---:|
{chr(10).join(rows)}

输出速度提升为baseline/candidate-1；耗时下降为1-candidate/baseline，详见summary.json，二者不混用。fresh两档各为三份固定输入中位数，缓存与持续输出单独列，后两项为单次。A/B/A都在127.0.0.1:19002隔离端口依次重新加载，任意时刻只有一个完整Engine；输入、CPU线程、图档位、权重源及其余优化参数相同。

以回切r152作较快对照，fresh 8k/长文每token分别节省{s['metrics']['median_8k_tpot_ms']['rollback']-s['metrics']['median_8k_tpot_ms']['candidate']:.2f}/{s['metrics']['median_256k_tpot_ms']['rollback']-s['metrics']['median_256k_tpot_ms']['candidate']:.2f}ms。相较r154约4.17ms的48层加权代理，这是本轮实际HTTP结果；本轮单项尚未达到输出速度+20%。

fresh 8k为8064输入+128输出，长文为262016输入+128输出，实际n_prefill分别为8060/262012，各阶段一致。每次核查n_prefill确保重新预填，而非把命中缓存算提速；128输出长度严格相同。1024是请求上限，实际输出数分别为{[s['phases'][p]['sustained_actual_tokens'] for p in ('baseline','candidate','rollback')]}，不能把不同实际长度误称为固定1024。该持续输出也覆盖长文之后返回短文的路径。

基线与回切逐请求回答一致性：{s['baseline_rollback_answers_exact']}。基线/候选/回切的长文三事实召回分别为{[s['phases'][p]['long_recall_hits'] for p in ('baseline','candidate','rollback')]}。这些不是完整质量验收，候选允许数值与回答变化。

## TTFT（秒）

| 场景 | 基线r152 | 候选 | 回切r152 |
|---|---:|---:|---:|
{chr(10).join(tt)}

HTTP TTFT取首个非空SSE文本；完整读取DONE和fnx_stats。TPOT用原服务的生成统计，同时以客户端首文本到结束的耗时除以后续token数交叉核对，见http-timing-crosscheck.json。启动/加载/图捕获、排队与HTTP TTFT不混入TPOT口径。三阶段启动时间见*-startup.json。所有正常基准均无profiler采集。额外恢复9002后的fresh 8k只作正式入口恢复烟测，不混入A/B/A统计。

## 候选确实生效

使用r154已通过开发筛选的固定α=.5 SmoothQuant W8A8，qkv与z合并；保留router、小mixer、KV和FP32递推状态的原路径。父/子进程实际装配审计记录48层、{sum(len(r['projections']) for r in audit)}个SmoothW8A8对象、两个进程；逐项验证INT8权重、BF16输入缩放及校准文件SHA。不是只根据环境变量判断开关生效。

private /fnx/loss 的16目标参考值为基线5.178334/候选5.154368。首次候选整模型值5.154514，相差+0.000146nat，被不适合数值变化候选的1e-5启动检查中断；自动恢复正式r152后保留initial-attempt原记录。候选启动烟测容差改为0.001nat，基线仍为1e-5；这只用于排除装配异常，不改变r154开发集平均0.02/最差切片0.05nat的筛选门槛，也不等同最终质量通过。候选重新加载后的16目标mean及逐token NLL与首次精确相同，见candidate-smoke-repeatability.json；与逐层参考的微小差异原因未在本次计时中定位。已完成基线A保留，继续重新加载候选B、回切A2；最终A/A2漂移同时报告。

r154开发集2048目标的平均ΔNLL={q['mean_delta_nll']:+.6f}nat、最差切片{q['worst_case_delta_nll']:+.6f}nat；该集已经用于开发筛选，不是最终holdout。完整任务、代码pass@1、多长度长文与多轮质量仍未验收，本次性能结果不能当成精度发布许可。

## 受控执行与恢复

停止前检查9002无连接、pipeline传输计数连续稳定，记录原服务/worker PID、出生时间和私有启动环境。暂停已识别watchdog与manager，唯一临时RefuseManualStart drop-in防止争抢。正式源码、ELF、p275入口均未替换。

服务端controller顺序执行A/B/A，EXIT自动恢复；独立45分钟恢复timer和监护进程覆盖controller退出、超时及客户端断链。只停止记录的原进程或本轮隔离目录里的诊断server/worker。通过原p275入口恢复9002、manager和一份正常watchdog，删除私有ENV和临时租约/禁止启动配置。最终源文件哈希、健康状态及辅助进程清理见final-operations.json。

本轮是临时独占实测，未将未完成全面质量验收的候选切入正式9002。原始目录/models/fnx-diagnostics/r155-smooth-tpot-ab-20261113。证据只包含脚本、摘要、审计和基准回答，不含模型权重、校准向量、图缓存、完整ENV或凭据。输入沿用r152固定清单，SHA见input-manifest.json。
'''
(P/'report.md').write_text(report)
r=s['metrics']['median_8k_tpot_ms'];l=s['metrics']['median_256k_tpot_ms']
(P/'techdoc-block.md').write_text(f'''## r155 整模型TPOT实测（2026-11-13，恢复正式r152）

r154 SmoothQuant W8A8候选做了完整HTTP A/B/A，未加入down GEMV。fresh 8k三次TPOT中位{r['baseline']:.2f}→{r['candidate']:.2f}ms，回切{r['rollback']:.2f}ms；输出速度相对基线{r['speedup_vs_baseline']*100:+.2f}%。fresh 262016输入三次{l['baseline']:.2f}→{l['candidate']:.2f}ms，回切{l['rollback']:.2f}ms；输出速度{l['speedup_vs_baseline']*100:+.2f}%。缓存及持续输出分列于报告，不混入fresh中位数。

48层96个实际SmoothW8A8投影对象、两进程和校准SHA均审计。r154开发NLL通过不代替最终生成/长文任务质量验收，候选未发布；测试后经原入口恢复r152，部署源码/ELF及启动配置不变。详见[r155整模型实测](r155-smooth-tpot-ab.md)。

''')
print('REPORT_RENDERED')
