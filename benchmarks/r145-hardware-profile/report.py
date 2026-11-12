import json,statistics
from pathlib import Path
P=Path(__file__).resolve().parent;s=json.load(open(P/'summary.json'));runs=json.load(open(P/'runs.json'))
def val(f,k):return f.get(k,{}).get('weighted_mean')
timing='| 上下文 | 无采集前 / 后, ms | timing采集, ms | 设备忙 | 级内间隙 | 外部间隙 | 阶段间空档 |\n|---|---:|---:|---:|---:|---:|---:|\n'
pmu='| 上下文 / 阶段 | Cube MAC周期占比 | Vector周期占比 | AIC MTE2周期占比 | L2读命中率 | HBM读 / token | HBM写 / token | HBM平均读 / 写 |\n|---|---:|---:|---:|---:|---:|---:|---:|\n'
over=[]
for c in ('8k','256k'):
 t=s['timing'][c];b=s['baselines'][c+'_baseline_before'];a=s['baselines'][c+'_baseline_after'];base=(a['mean_ms']+b['mean_ms'])/2
 timing+=f"| {c} | {b['mean_ms']:.3f} / {a['mean_ms']:.3f} | {t['host_period_ms']:.3f} | {t['device_busy_ms']:.3f} | {t['intra_stage_gap_ms']:.3f} | {t['outer_gap_ms']:.3f} | {t['handoff_gap_ms']:.3f} |\n"
 over.append(f"{c} timing扰动约 {(t['host_period_ms']/base-1)*100:.2f}%")
 hw=s['hardware'][c]
 for st in ('0','1'):
  f=s['pmu'][c]['PipeUtilization']['stages'][st]['total']['fields'];l2=s['pmu'][c]['L2Cache']['stages'][st]['total']['l2_read_hit_rate'];h=hw['devices']['NPU '+st]
  pmu+=f"| {c} / S{st} | {val(f,'aic_mac_ratio')*100:.2f}% | {val(f,'aiv_vec_ratio')*100:.2f}% | {val(f,'aic_mte2_ratio')*100:.2f}% | {l2*100:.2f}% | {h['hbm_read_MB']/hw['steps']/1000:.3f} GB | {h['hbm_write_MB']/hw['steps']/1000:.3f} GB | {h['hbm_read_MB_s']/1000:.1f} / {h['hbm_write_MB_s']/1000:.1f} GB/s |\n"
clocks='; '.join(f"{r['context']} 最小往返 {r['best_rtt_us']:.2f} µs，子进程相对中点偏移 {r['child_epoch_offset_from_midpoint_us']:.2f} µs" for r in s['clock_alignment'])
text=f'''# r144 双进程对齐时间线与硬件计数器（r145 补采）

2026-11-12，192.168.18.206，2×910B4-1，物理3/5、容器0/1。采样实现为794a22e的r144，正式模型代码、权重、共享库、启动参数均未改。诊断副本仅给pipeline/worker加入计时范围和profiler RPC。BF16激活/KV、FP32状态，24+24两进程，262144窗口，hybrid传输和融合head保留。

## 时间线账本

{timing}

时间单位均为每token毫秒。设备忙为AI Core/AI Vector/Memcpy区间的并集；级内间隙为两阶段包络并集减设备并集；外部间隙为token周期减阶段包络。三项严格闭合，跨卡执行重叠为{s['timing']['8k']['cross_device_overlap_ms']:.6f} / {s['timing']['256k']['cross_device_overlap_ms']:.6f} ms。不把两卡利用率相加当单卡利用率，也不把等待另一阶段的空闲误判为Host喂数不足。

每个timing轮24步，按相邻完成时刻计23个周期；前后无profiler基线各16步。{'；'.join(over)}。这些是直接Engine.generate驱动的runtime周期，含生成控制，但不含HTTP/SSE文本编码流程；与上一轮HTTP TPOT 33.78 / 34.45 ms口径不同。计数器轮8步，性能受PMU采集影响，不充当正常TPOT。

Host源仍有输入准备、图选择/回放、D2H同步、Pipe交接、H2D同步、greedy标量读取及返回。表中阶段间空档位于stage0最后设备任务与stage1第一设备任务之间；D2H/H2D设备任务本身另计设备忙，因此此列不是完整通信总耗时。CPU的transfer_wait/recv_wait/greedy_wait范围包含NPU执行等待，不能重复加到设备时间上，也不能都算CPU运算。

对齐依据：同机epoch与monotonic时钟、两进程共同token序号、CANN绝对时间。{clocks}。没有为了闭合人工移动或缩放设备时间。coverage-audit.json检查各周期kernel签名数量一致、HBM时间窗覆盖和阶段先后关系。

`aligned/8k_timing.json`、`256k_timing.json`为完整Chrome Trace格式，包含两进程CPU范围与两卡kernel；`*_two_tokens.json`是两token窗口。`r144-dual-process-timeline.png`用于快速查看；灰色Host范围包含等待，不能按面积当作CPU忙碌。

## Cube / Vector、缓存与物理HBM

{pmu}

以上各指标分轮采集，不是同一瞬间的联合计数。MAC/Vector/MTE2按对应Core total_cycles加权；它们可重叠，不能相加到100%。这是参与核的指令周期占比，不是整卡峰值FLOPS达成率。CANN的cube_utilization列是核心周期覆盖口径，不能当成MAC指令占比或BF16峰值利用率。

L2读命中率按aic/aiv的r0/r1命中计数总和除以命中+miss_allocate计数总和，保留原始计数与覆盖率，不采用各算子百分比简单平均。MemoryAccess逻辑数据量和Memory组带宽在summary.json中按算子类别列出；不能将GM读量直接当作穿透L2后的物理HBM字节。

物理HBM与PCIe使用单独的msprof系统采样：HBM请求100Hz、PCIe请求50Hz。只积分实际32个decode token对应的时间窗，排除工具启动、导出和空闲段；按导出单位MB/s积分。每卡四个HBM通道相加。工具hbm.csv的Average是四通道平均值，不能直接当整卡总带宽。读写相加也不能与纯读峰值直接比较。系统轮不同时运行算子PMU，避免采集资源争用。

本机历史连续HBM读基线约1447 GB/s。表中平均带宽包含该卡等待另一阶段的时间；低于这个基线并不排除特定kernel内的访存瓶颈。结合低MAC指令占比、较高MTE2占比与大量短kernel，当前证据更支持“数据搬运等待、访问/执行粒度效率”限制，不能表述为Cube算力跑满或HBM总线持续跑满。PCIe在hybrid decode中搬运约20KiB隐状态/阶段边界，固定交接延迟需要优化，但没有大流量传输饱和的证据。

## 采样与恢复

固定使用r144已验收的8064及262016输入token的两份HTTP消息，Engine.generate产生124个连续token；不复制整个256k缓存回滚。每轮位置在runs.json列明，未越过窗口。两轮输出与r144参考前缀一致性记录于8k/256k-output.json，未修改数值路径。

最初8k全部完成后，256k输入文件名不一致触发中断；独立恢复任务已实际恢复正式r144。修正入口后只续采256k，并保留8k数据，没有把失败启动算成性能结果。

正式服务在采样时暂停，manager和两份已确认的FNX watchdog停止。独立systemd到期恢复覆盖SSH断链；完成后通过原p275.sh重新HTTP ready，恢复manager并还原为一份正常watchdog，移除临时drop-in、恢复timer、租约及本轮完整ENV快照。所有诊断改动都在隔离副本，正式源码/ELF与采样前哈希核对。

原始目录：/models/fnx-diagnostics/r145-hardware-profile-20261112。完整traces和system原始文件保留服务器，不把权重、图缓存、完整ENV或凭据放入Git/交付包。profile_support.py、patch.py、profile_model.py及analyze.py可复核方法；输入文件、软件矩阵和正常服务配置须一致。硬件计数器解释依据CANN9.1的AiCMetrics、op_summary与系统HBM/PCIe字段，不把旧r141占比外推至r144。
'''
(P/'report.md').write_text(text);print('REPORT_READY')
