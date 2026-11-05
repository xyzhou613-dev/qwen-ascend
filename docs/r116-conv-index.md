# r113–116 卷积与 Indexer 索引整理优化

2026-11-05，2×910B4，48 层 24+24，256k，8192-token 主块，hybrid IPC，BF16 KV 和 FP32 DeltaNet。

刷新 r112 的设备采集后，256k 后段两阶段的稀疏注意力约占 13%，Top-K 类算子约占 5.7%。decode 仍包含大量小矩阵调用、Gather 和转置。比例为 kernel 累计设备时间，采集带排空和 profiler 扰动，作为定位证据。

卷积融合历史拼接、逐项乘法、FP32 成对归约及历史更新，保留原 SiLU。每个 BF16 乘积先舍入，再按 (a+b)+(c+d) 累加。包含消减误差的测试排除了其他归约顺序；256 步动态回放和真实 L00 的卷积、递推状态及输出逐位一致。

Indexer 保留现有评分与 Top-K，排序 512 个选中组后在原生内核展开，并按原规则补齐尾部、无效位置和物理顺序。四种行数和九组位置的动态重放，indices 与 mask 逐位一致。

| 256k 测例 | TTFT | Prefill | TPOT | 输出速度 |
|---|---:|---:|---:|---:|
| 同机 r112 基线 | 90.265 s | 88.940 s | 51.11 ms | 19.57 tok/s |
| r116 候选 | 88.434 s | 87.077 s | 49.40 ms | 20.24 tok/s |
| r116 生产复测 | 88.286 s | 87.009 s | 49.03 ms | 20.40 tok/s |
| 候选持续生成 | 87.469 s | 86.148 s | 49.25 ms | 20.30 tok/s |

同长度比较采用全新 262016 输入和 128 输出。候选持续测例为 261120 输入、1024 输出。三位置检索 3/3，重复请求无新增 prefill，追问通过。NLL 使用冻结参考 2.164792±0.01。

构建使用 kernels/conv/build.sh 和 kernels/index_expand/build.sh，目标为 dav-c220-vec。共享库与对应 fnx 模块放在同一目录。FNX_CONV_NATIVE 和 FNX_INDEX_EXPAND 默认关闭，生产通过 prod_pipeline.sh 启用。关闭对应开关并重启可逐项回退；本机整轮回滚为 bash /root/ascend-dev/r116/rollback.sh。

根分区满曾中断采集。已逐进程核对旧流水线缓冲的映射和命令行引用，将约 5.5 GiB 无引用文件迁移至 /models/fnx-diagnostics/inactive-pipeline-buffers-20261105 归档，保留原文件。新 trace 写入模型盘，原始 trace 留在本机 /models/fnx-diagnostics/r113/traces，Git 保存汇总和区间记录。

单次基线、候选和生产复测用于确认本轮收益，持续测例作为独立长度的稳定性检查。
