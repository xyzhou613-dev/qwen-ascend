# r157：整模型组合对照与完整质量配对

生产入口仍为 `/root/ascend-dev/p275.sh` 的 r152。候选只在独立私有目录、单个 Engine、localhost:19002 运行；同一时刻不保留另一份完整模型。

## 对照

- baseline：r152。
- smooth：r154 固定 alpha=0.5 的 SmoothQuant W8A8，96 个投影；不是重新校准。
- combined：smooth 加 r156 原 expert-major 布局 MoE INT4 down 与对角 epilogue，两 kernel 版本。
- rollback：r152 再测，并提供完整质量配对基线。

每阶段分别执行 fresh 8k 三次、fresh 262016 token 三次、长文缓存请求和持续输出。HTTP TPOT、TTFT、启动到 HTTP ready 分开记录，缓存结果不计入 fresh。完整层生命周期审计验证两个进程共 48 层、96 个 int8 投影；组合路径必须在全部 48 层命中。

这里保存的是本机本轮实际执行的脚本，路径和恢复入口有意固定。`controller.sh` 是保留已完成 baseline 后的续跑版本，依赖 `baseline/summary.json`；初次执行脚本与评分失败记录另存于 initial-attempt。不要把续跑脚本当作新环境的一键启动器，也不要在重试时删除 window-start.txt 重置截止时间。重新复现实验需要在新的独立目录准备同一来源、输入与基线，并保留同样的限时恢复约束。

## 冻结质量输入与验收

在组合候选执行前冻结 `acceptance-policy.json` 和输入哈希。r156 的官方源 revision、原始文件哈希及制备脚本见相邻实验目录；原始任务、答案、测试代码及 token IDs 留在 server-only private-quality，不放入证据归档。

1. 196 段、100352 个 NLL 目标；中英知识、数学、代码各 49 段。均值增量上限 0.02；预先固定 28 个切片（每领域按冻结次序分为 7 组，每组 7 段），切片上限 0.05。
2. CMMLU 300、MMLU-Pro 300、GSM8K 200、HumanEval 100、MBPP 100、合成严格 JSON 200，共 1200 题。新增的 100 道结构化任务也在候选运行前冻结。
3. 任务按 cmmlu/mmlu_pro/math/code/json 五域检验，2 个百分点非劣界值、单侧 95% 置信度。以“改善率的 CP 下界减退步率的 CP 上界”作保守配对下界；同时估计两个边界时分配 alpha。零改善时改善下界固定为零。子基准点估计退步超过 2 个百分点还需复核。
4. 新鲜中英文文学自然文本各 8k、32k、64k、128k、262016 token，使用真实分块 prefill，每段计分末尾 199 个目标，共 1990 个；每切片 NLL 增量上限 0.05。旧 r152 五个自然文本切片另作回归覆盖，不称为新 holdout。

`quality_core.py` 显式记录 token 对齐，计分完成或异常均清空递推状态。worker 内 CPU log_softmax 的评分必须与原 parent CPU-logit 口径逐 token 相符，最大绝对误差小于 0.00005；r152 pilot 均值还须满足原参考值。第一次 NPU log_softmax 评分超出这个门槛后被拒绝并恢复服务，保留失败证据，没有放宽阈值。CPU 评分只用于质量路由，不参与 HTTP TPOT 测量。

代码题在一次性 CPU 容器中执行，禁网、只读、非 root，并限制 CPU/内存/PID/时间；不在模型容器或宿主直接执行生成代码。检查点逐项 fsync；仅续跑客户端可修复自己文件的中断尾行。进度查看及分析脚本只读检查点。

`analyze.py` 对不完整配对输出 `incomplete_no_acceptance`，不能把部分结果当成通过。公共题目可能进入过预训练；JSON 提取不代表原生 tool-call 协议；文学 prefill NLL 不代表所有长文问答能力。W8A8 单独版本仅有短筛查，不能据此自动认定其完整质量通过。

## 时间边界与恢复

用户授权本轮完整配对最多六小时，超时恢复并保留断点。窗口从 2026-11-13 18:24:53 CST 开始；重试沿用该时间，不能重新计时。23:59:53 触发恢复，距 2026-11-14 00:24:53 上限留出 25 分钟恢复余量。controller EXIT trap、独立 systemd timer、独立 guardian 均可触发同一带锁恢复脚本。恢复原入口、8099 manager、单份正常 watchdog，移除实验 lease、私有进程 ENV 和临时 drop-in。

后续又增加 23:59:20 的独立调度（restore-dispatch.json），提前 33 秒启动同一个恢复 service，避免恢复命令处于正在被 RuntimeMaxSec 终止的 controller 控制组。原保护未延长或移除；该临时调度 timer 也须在恢复后清理。

实测 daemon-reload 会重算本次 transient OnActive 相对 timer，因此最终两道恢复 timer 改为固定 UTC OnCalendar 绝对截止、AccuracySec=1s，配置副本随证据保存。独立 dispatcher 先在恢复锁内终止本轮评分客户端和所属 unit，之后才启动恢复 service 清理代码判题容器；避免将中断的判题记为模型错误。新增 service/timer drop-in 同样在恢复后清理。

主机重启后的原容器设备节点修复另由持久的 `ops/fnx-boot-restore` 提供。这是保留的正常恢复服务，与本轮需清理的临时实验 timer 不同。主机重启后不自动续跑实验。

## 可选硬件采样

仅在完整质量配对后且预算充足时，用单 Engine 分别采 smooth/combined 的双进程时钟对齐、设备时间并集、独立 PMU passes 和系统 HBM/PCIe 样本。`analyze_profile.py` 保留原始计数单位与覆盖率：GM/main-memory 流量不等于物理 HBM；MAC/Vector 指令周期占用不等于峰值 FLOPS；进程等待包含设备执行，不能与设备耗时重复相加。跨阶段间隙也不能直接当作 IPC 的物理传输耗时。

结果以最终报告、成对分析和最终恢复验证为准。源码与指标可复核不代表候选已经发布。
