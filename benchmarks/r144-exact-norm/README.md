# r144：精确归一化、DeltaNet 复用及 Raw 合并否决证据

正式实现为 `fnx/norm_post.py`、`kernels/l2norm_post/` 与 `engine.py` 的两个独立开关：`FNX_NORM_POST=1 FNX_DELTA_REUSE=1`。源码默认关闭。Raw 全量合并与仅 decode b+a 均因真实分块 prefill 评分回退被拒绝，正式 engine.py 没有这些修改。详细结论见 `docs/r144-exact-norm.md`。

`summary.json` 保存完整 HTTP 指标、质量摘要与发布源文件 SHA256。8k 文件夹每组3次 fresh 输入，8064+128；256k 每组1次 fresh 输入，262016+128，再做独立缓存重复。HTTP TTFT/端到端与引擎 TPOT 分列，`repeat_262144.json` 的新增 prefill 必须为0，不能作为 fresh 收益。没有新的完整硬件 profiler 账本。

## 质量和组件证据

* `fixtures.json`：冻结6k自然文本、12k重复服务文本、24k技术文档的 token IDs；12k属于较容易的重复输入。每份尾部199个next-token目标，baseline/candidate-quality.json保留逐目标NLL、argmax与IDs哈希。
* `prefill_nll.py` / `quality_route.py`：走 plan_prefill、prefill_chunk、finish_prefill；完整模型诊断使用同一个已加载 Engine。临时路由必须放在 `__main__` 启动语句之前，且从不安装到正式服务。
* `norm-probe.json` / `norm-extremes.json`：真实权重投影→BF16卷积/SiLU输入、全部非负有限BF16求和位型及+Inf、18组正负零/极小值/溢出案例。保留BF16平方/ReduceSum和舍入边界。前者的prefill微测是连续布局筛选；正式只启用单token后处理。
* `delta-probe.json`：T=1/63/64/65/128/512/2048，连续三次状态递推，输出BF16与状态FP32逐位比较；integrated_delta_probe.py验证实际集成开关。
* `full-layer.json` / `full-layer-bits.json`：真实L00/L03权重，多步T=1/128/512/2048，输出/FP32状态/BF16卷积历史/原始及packed KV/索引位型检查。对拍包括正负零；bits组只做精度复核。
* `probe.json` / `ab-probe.json` 和 `rejected-*`：Raw真实shape设备微测、真实层及全模型失败结果。图计时在图内重复，并在精度对拍前显式replay；初版未replay与启动开销占主导的数据已弃用。

## 独占与重现

不要在9002正式模型占卡时另起完整Engine；单层实验也可能超过剩余HBM。prefill_nll.py拒绝第二份完整模型，交付的full_layer_probe.py也增加了存活模型检查。使用本机完整正式参数、相同dtype/权重/分块策略，不要只设置两个新开关就当作同配置测试。production-selected-config.json只保存非敏感的模型参数与CPU设置。

在容器内放置内核源码后，用 `bash build.sh /明确输出路径/libfnx_norm_post.so` 编译，停使用进程后安装。独立探针可使用正式模块目录，例如 `PYTHONPATH=/root/ascend-dev/fnx`；若从仓库导入wrapper，设置 `FNX_NORM_POST_SO` 指向已验证共享库。这里的探针计算逻辑与现场相同，交付副本只调整模块导入位置和单层实验保护。

controller/rescue/deployment脚本记录本机独占、快照、限时回退和正式入口验收流程，路径固定为本次206证据目录，迁移前应审查路径与服务归属。完整ENV快照只在受保护诊断目录短期存在，结束后删除，未包含在Git或附件。脚本源文件中的引用不是实际私有环境。正式配置启用器为 `scripts/r144-enable.py`，支持 `--check`。

watchdog恢复须独立于实验控制器cgroup；现场最终还原为原来由SSH入口启动的正常watchdog，未遗留试验守护单元。历史100题、27B来回切换和MTP本轮未重跑。
