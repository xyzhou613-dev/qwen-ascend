# r157 后续方向的约束记录

完整质量配对尚在运行时记录；不代表候选验收通过或任何下列方向已经取得收益。

本轮正常 HTTP 已测得 W8A8 8k/长文 29.38/30.07 ms，组合版 27.96/28.54 ms，初始 r152 33.01/33.63 ms。新 down 在全部 48 层命中。组合版相对 W8A8 的实际额外节省为 1.42/1.53 ms，不能再叠加 r156 的整层代理。完整 A/B1/B2/A 和质量结论以最终报告为准。

优先完成本轮配对。若质量不通过，先按冻结切片定位误差、区分 W8A8 与 down；W8A8 单独版本也不能靠 2048 目标筛查替代完整验收。不得根据候选结果重新挑选本轮切片或修改门槛。

下一轮可核查 MoE up 的 A8W4 分组矩阵乘、SwiGLU 和量化融合。官方接口列出 A2 产品及 per-group A8W4，现场容器也有对应头文件和 libopapi 导出符号；这些都不能证明当前每专家单 token 的性能、动态路由图捕获或质量已可用。需要先真实 shape 小测，改变专家 IDs 后检查 graph/eager，计入路由、量化、scale 编码、权重选择和后续反量化的全部开销。接口约束见 [GroupedMatmulSwigluQuant](https://www.hiascend.com/document/detail/en/canncommercial/850/API/aolapi/context/ops-transformer/aclnnGroupedMatmulSwigluQuant.md) 与 [官方源码目录](https://gitcode.com/cann/ops-transformer/tree/master/gmm/grouped_matmul_swiglu_quant_v2)。

一个实际容量限制：当前 BF16 per-group scale 换成接口使用的 UINT64 scale，若为两种 MoE 投影全量常驻且丢弃旧 scale，24 层每卡会额外增加 5400 MiB；保留两份则增加 7200 MiB。本轮组合版做过长文后的 npu-smi 空余仅约 6093～6263 MiB。这个快照包含 allocator 保留量，不是精确可用峰值，但足以否决“无需预算直接全部常驻”的做法。必须测 allocated/reserved 和 prefill 峰值，评估只改 up、受限专家缓存或在原布局直接消费 BF16 scale；不能宣称换 INT8 就会降低总内存。

MTP 继续复用 [r153 已有验证器](../../docs/r153-tpot-directions.md)：T4 四层加权代理从约 131ms 降到 79.94ms，含小 T FP32 recurrence 前缀 journal，尚缺 head、IPC、drafter 和完整 rollback。下一项将当前量化/数值修正纳入同轮 V4 重测并补齐这些成本，不能把 M1 节省直接从旧 80ms 相减。若未来在 27.96ms 骨干上再提高 20% 输出速度，假设 k=3、p=.8，完整周期需≤68.8ms；这是条件预算，当前候选尚未通过质量。收益用“实际接受 token 数 / 草稿、验证、状态恢复总时间”计。检查点覆盖 FP32 DeltaNet、conv、KV/index 尾部及位置，避免复制整个长文缓存；分块 prefill 与 decode 的数值路径差异仍需验证。

TP2 仍作备选：将 24+24 流水并行改为两卡张量并行，使同一 token 的大投影同时使用两卡。现场 `npu-smi info -t topo` 再确认 3/5 为 SYS、跨 NUMA、未显示 HCCS；当前两个进程 CPU 均允许 0-63、内存节点 0-1。r153 已在对应 NUMA 绑定下验证 HCCL 图捕获和非零数据求和：5/10/20KiB 的图内 all-reduce 约46～51µs，96次约4.7～4.8ms。该历史实测应加入预算；只在设备/软件配置变化时复核。下一项是完整真实 shard 层及两次 all-reduce 的净收益。[HCCL AllReduce 官方接口](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/commlib/hcclug/docs/zh/api_ref/comm_op_interface/HcclAllReduce.md)列出 A2 支持，不能据此宣称 TP Engine 已实现。

TP 先设停止条件再写完整 Engine：测真实投影与 MoE 半尺寸 tiling，并包含已有通信成本。粗略预算为 `T_TP ≈ D + C/2 + 96×L + R`，其中 D 是不能分摊的时间，C 是可分摊部分，R 是 tiling/路由新增成本。仅举例：以27.96ms、D=5ms、R=0，要达到1.5×速度，L需低于约22.5µs，低于r153约46～51µs的实测；这说明直接标准TP2没有这样的收益依据。D尚未由本轮新PMU测得，公式不是预测结果。长文还要核算大payload通信量、KV/DeltaNet分片及显存峰值。微测预算不成立时停止完整TP改造。

本轮可选 PMU 用于决定余下预算应投向哪一项。HBM 采样、GM/L2 计数、Cube/Vector 指令周期与 Host/IPC 等待分开解读；不把 npu-smi AICore 百分比当峰值计算效率，也不把跨进程等待都归因于互连带宽。
