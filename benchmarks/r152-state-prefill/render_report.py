from pathlib import Path
import json

def render(P,R,outcome):
 r=json.load(open(P/'ab-summary.json'));b=r['baseline'];c=r['candidate'];deployed=outcome['status']=='deployed'
 prod=json.load(open(P/'production/summary.json')) if (P/'production/summary.json').exists() else None
 measured=prod if deployed else c
 table='| 指标 | 同轮r151 | 隔离候选 | 正式复测 | 变化 |\n|---|---:|---:|---:|---:|\n'
 for label,key,kind in [('8k TTFT中位（s）','median_8k_ttft_s','ttft'),('256k三次TTFT中位（s）','ttft_256k_s','ttft'),('8k TPOT中位（ms）','median_8k_tpot_ms','tpot'),('256k三次TPOT中位（ms）','tpot_256k_ms','tpot'),('缓存256k TPOT（ms）','cached_256k_tpot_ms','tpot')]:
  change=100*(1-measured[key]/b[key]) if kind=='ttft' else 100*(b[key]/measured[key]-1)
  formal=f'{prod[key]:.3f}' if prod else '—'
  table+=f'| {label} | {b[key]:.3f} | {c[key]:.3f} | {formal} | {change:+.2f}% |\n'
 qb=json.load(open(P/'baseline-quality.json'));qc=json.load(open(P/'candidate-quality.json'));qrows=''
 for name,x in qb.items():
  y=qc[name];nd=sum(a!=b for a,b in zip(x['nll_tokens'],y['nll_tokens']));ad=sum(a!=b for a,b in zip(x['argmax'],y['argmax']))
  qrows+=f"| {name} | {x['tokens']} | {x['scored']} | {x['mean_nll']:.6f} | {y['mean_nll']:.6f} | {nd} | {ad} |\n"
 events=[]
 for d in (0,1):
  long=json.load(open(P/f'long-graph-device{d}.json'))['results']['8']['per_update_us']
  f=json.load(open(P/f'delta-function-v3-device{d}.json'))['records'][0]['timing']['median_us']
  events.append(f"卡{d}长图每次状态更新{long['torch']:.3f}→{long['native4']:.3f}µs；512-token完整函数{f['torch']:.1f}→{f['native4']:.1f}µs，私有复用版本{f['inplace4']:.1f}µs")
 service='r152' if deployed else 'r151'
 decision='已正式发布r152，并在9002复测通过。' if deployed else '未采纳候选，正式服务为r151。'
 startup=[json.load(open(P/f'{v}-startup.json'))['start_to_http_ready_s'] for v in ('baseline','candidate')]
 report=f'''# r152 FP32 prefill状态更新：{'正式验收' if deployed else '候选否决记录'}

2026-11-13，192.168.18.206，2×Ascend 910B4-1，物理3/5、逻辑0/1，24+24两进程。CANN9.1.0、驱动26.1.1、torch2.10.0+cpu、torch_npu2.10.0.post4。起点b75d63f / r151，仓库干净。

{decision} 结论：{outcome.get('reason','精度和性能门禁通过').rstrip('。')}。{'正式启用FNX_STATE_UPDATE=1，保留r150归一化与r151输出融合。' if deployed else '不能将组件收益算作已交付的整模型提速。'}

## 整模型HTTP结果

{table}

TTFT变化是`1-新TTFT/基线TTFT`，TPOT变化表示输出速度`基线TPOT/新TPOT-1`。{'变化列采用正式复测。' if deployed else '变化列仅表示未发布候选。'} 8k为8064输入+128输出，每版本三份固定输入取中位。长文为262016输入+128输出，每版本三份不同文档标识的输入，核查新增prefill确保fresh；不能外推长输入追加8k输出。缓存命中新增prefill=0，采用非流式接口，单列。

长文基线三次TTFT（s）：{b['ttft_256k_runs_s']}；隔离候选：{c['ttft_256k_runs_s']}；正式复测：{prod['ttft_256k_runs_s'] if prod else '无'}。预设性能门禁：三次TTFT中位降低>0.5%，各次均降低；8k TTFT及两场景TPOT不回退>1%。先验收995个质量目标和全部回答精确一致，再判断性能；正式入口必须重现收益。

基线和候选均重新加载，使用相同配置、权重和冻结输入，在仅本机127.0.0.1:19002测量，每次只有一个完整Engine。暂停已识别watchdog与manager，独立恢复timer保护断链和超时。HTTP ready分别{startup[0]:.1f}/{startup[1]:.1f}s，启动/图捕获不算请求TTFT。隔离结束先经原入口恢复r151，再决定是否发布。

## 内核与测量修正

仅在delta_chunk_batched_fp32中，将`current*end_decay+state_delta`的两次FP32运算合并到单个AscendC kernel。保持FP32乘法的舍入后再相加，使用Muls、PipeBarrier、Add，不使用FMA；不把递推状态转BF16。state_delta仍由同一个BMM产生，不改BMM输入、形状或结合顺序。乘法移到该独立BMM之后，数学依赖保持；FP32状态与最终输出逐位对拍。

默认mode4，TILE4096：48个FP32 decay先整段搬入UB，每核只加载一次；输入与输出各双缓冲，用队列事件衔接MTE2、Vector、MTE3，替代每tile的完整串行栅栏。形状固定为FP32 current/delta `[48,128,128]` 和decay `[48,1,1]`，同卡连续；形状不符回原运算。返回新状态，不修改调用者初态。decode与小于等于64的独立分支不启用此内核。

每次状态更新的逻辑张量读写约15MiB→9MiB，省6MiB中间读写；这些访问可能命中L2，不能把逻辑量当作物理HBM流量或直接推算TTFT。

最初8次更新的短图，各版本都约0.29–0.30ms，无法分辨单次内核成本。改成256次更新，循环复用有限缓冲，避免大输出列表和重放成本主导。没有补采Host trace，因此不把短图差额全归为Host调度。完整分块函数使用合成q/k/v和FP32初态，不包含投影与MoE；真实权重层检查另列。长图与整函数结果：

- {events[0]}
- {events[1]}

七种配置、双卡各64组测试涵盖不同幅值、极小值、相消、decay0/1及显式FMA反例，逐位通过。固定私有状态缓冲区需要先clone输入，完整函数没有更快，未采用；源实现保留为否决变体。

完整函数探针首版同时保留五份图，触及自身约1GiB allocator上限，当时物理设备仍有3.76GiB可用，生产服务健康检查通过。未扩大上限；改为只保留必要图后完成测试。保留初次报错及恢复后健康记录，不能把该错误描述为正式模型OOM。

## 精度验证

全部36个真实权重DeltaNet子路径，65/128/512 token eager及4次128-token动态graph replay，每层7例共252例；输出、BF16卷积历史和FP32状态逐位一致。L00/L12/L24/L44完整层decode与128/512/2048 prefill也通过。

| 冻结文本 | token数 | 目标数 | 基线NLL | 候选NLL | NLL不同目标 | argmax不同目标 |
|---|---:|---:|---:|---:|---:|---:|
{qrows}

真实plan_prefill/prefill_chunk/finish_prefill，五份共995个尾部预测目标；原12k是重复服务文本，18k/36k为不重复填充的自然文本，来源/ID哈希沿用r150/r151清单，历史100题未重跑。三个fresh 8k、三个fresh长请求和缓存回答一致性见ab-summary.json；长请求分别要求三事实召回3/3。样本通过不构成所有输入等价的证明。

## 最终状态与交付

最终健康版本{service}，窗口262144。manager active、RefuseManualStart=no，正常PPID1 watchdog一份。临时timer/drop-in/租约、诊断模型和完整ENV私有快照清理，完整状态见final-operations.json。{'五个发布文件有精确备份和哈希保护，其余r151源码/ELF保持不变；正式server没有质量诊断路由。源码在kernels/state_update，Linux aarch64 ELF在implementation，启动补丁scripts/r152-enable.py可--check。设置FNX_STATE_UPDATE=0后须重启并重新捕获图，才能恢复原路径。' if deployed else '候选不在最终正式路径中，全部原运行文件与启动脚本恢复为测试前哈希。'}

证据白名单打包，不含权重、图缓存、完整ENV、凭据或用户激活。原始目录/models/fnx-diagnostics/r152-state-prefill-20261113。
'''
 (P/'report.md').write_text(report)
 block=f'''<!-- r152-current -->
## 2026-11-13 r152 FP32状态更新：{'正式发布' if deployed else '未采纳，生产仍r151'}

{decision} {outcome.get('reason','')}

{table}

8k三次中位，256k为262016输入+128输出、三次fresh中位。详见[r152实测](r152-state-prefill.md)与benchmarks/r152-state-prefill/。以下为历史记录。
<!-- /r152-current -->
'''
 (P/'techdoc-block.md').write_text(block)
 doc=R/'docs/Qwen3.8-Flash-Next-2x910B4-技术文档.md';s=doc.read_text();assert '<!-- r152-current -->' not in s
 head,rest=s.split('\n',1);doc.write_text(head+'\n\n'+block+'\n'+rest.lstrip('\n'))
 (R/'docs/r152-state-prefill.md').write_text(report+'\n[完整证据](../benchmarks/r152-state-prefill/README.md)。\n')
