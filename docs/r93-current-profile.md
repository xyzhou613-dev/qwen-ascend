# r93：当前双进程流水线的算子采集

2026-11-02，基于 r92 / 00a181fb，固定两卡910B4、W4专家、BF16 KV、FP32 DeltaNet。
两阶段分别采集真实8064输入的prefill、其前16步decode，以及262016输入约196611位置开始的两个8192块、其前16步decode。三位置事实7319/8652/2946全部检索正确。

为限制trace体积，采集仅覆盖上述区间；开始/结束会排空流水线，且本轮关闭对话快照以留出profiler内存。events.json是带采集扰动的区间时间，不能当作新TTFT、TPOT或吞吐成绩。以下占比是每卡设备kernel累计时长，不是完整请求墙钟占比；不同kernel可能在stream间重叠。

| 区间 | 两阶段观察 |
|---|---|
| 8k prefill | GroupedMatmul约43.1% / 43.6%；ScatterUpdate约7.5% / 7.4%；SparseFlashAttention约6.4% / 6.4% |
| 256k后段prefill | SparseFlashAttention约11.4% / 11.3%；ScatterUpdate约10.8% / 10.7%；GroupedMatmul约9.9% / 10.2%；ReduceSum约9.0%；BatchMatMul约7.9%；TopK约5.0% |
| 256k decode | MatMul约27.4% / 25.6%；Mul约10.7% / 10.3%；GroupedMatmul约8.7% / 8.1%；IndexSelect约6.7% / 6.1%。第二阶段包含输出头，WQBM约10.7%，第一阶段约4.8% |

这支持优先检查indexer缓存和评分、稀疏注意力，以及DeltaNet与MoE的小算子/布局。P2P传输不是当前主要计算热点；上述聚合尚不能把同名kernel精确归属到单个模块，也不能据此宣布达到硬件极限。后续内核修改需单模块trace和完整HTTP测例验证。

脚本在experiments/r93，汇总、区间、原始日志、配置和trace清单在benchmarks/r93。大CSV保留在服务器 /root/ascend-dev/r93/traces；清单记录路径、大小和SHA256。CANN图内kernel的Type列为N/A，汇总改用Name，避免误把全部算子归成一类。生产性能仍使用r92独立HTTP数据。
