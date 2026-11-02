# r94–95：Indexer去重与双KV快照瘦身

2026-11-02，r93热点证据提交1ecedfd，基线r92/00a181fb。保持24+24独立进程、262144窗口、8192主块、hybrid IPC、BF16 KV与FP32 DeltaNet；不改变模型权重、因果mask、attention预算或MoE数学路径。

## 结果

模型已加载，服务器本机HTTP，全新随机编号文档，262016输入、128输出，新prefill262012。生产复测：

| 次数 | 客户端TTFT | 引擎prefill | TPOT | 输出速度 | HTTP端到端 |
|---|---:|---:|---:|---:|---:|
| 1 | 102.048 s | 100.784 s | 54.05 ms | 18.50 tok/s | 108.913 s |
| 2 | 101.552 s | 100.313 s | 53.96 ms | 18.53 tok/s | 108.406 s |

r92生产TTFT为102.114/101.607秒，TPOT54.17/54.16毫秒。当前样本数量有限，不能据小幅波动宣称整请求显著加速。三次新8064-token输入的生产TTFT中位数5.872秒，原始逐次记录见benchmarks/r95/production8k.json。

## 实施与取舍

- FNX_INDEX_UNIQUE=1：固定形状计算触及的4-token组，避免每个token重复池化和RoPE。上界覆盖所有起始对齐，额外行夹到最后触及组，不写未来组；无CPU读pos、动态unique或新数值近似。140组边界及动态位置图重放通过，缓存、所选token和mask逐位一致。8192-token indexer图复测102.235→101.868 ms；这是组件小收益。
- 分组先排序再展开的候选逐位正确，但8192-token图约139.8 ms，明显慢于原始2051-token排序。未部署；复现候选和原始数据保留在experiments/r94及benchmarks/r95。
- FNX_STATE_DUAL_SLIM=1：新快照标记original-only-v1，仅保存原始BF16 K/V，加载时分8192位置用copy/zero恢复latent和RoPE。运行时双KV地址和图保持不变。旧完整快照、主机转换、save_into及compact长度变化兼容；不改变FP32 DeltaNet/conv/index状态。
- 全长两卡12个attention层快照少保存6.75 GiB。最新生产槽长度262044，实测快照7.2307 GiB；按实际长度确定省去的packed张量6.7474 GiB，等价旧格式13.9781 GiB。这是快照空间减少，并非持久KV、模型权重或总HBM占用减少同样数值。恢复需要重建工作，缓存命中TTFT可能增加；不把省内存直接当作加速。

## 验收与运维

- 隔离单元测试覆盖0/1/3/4/7/31/8192/262144长度、旧完整格式、主机快照和槽复用；满容量测试用随机BF16，另在8k容量显式覆盖全部65536个位型，重建缓存均逐位一致。满容量单元模型只有两组各1层KV，字节节省为1.125 GiB，12层生产窗口推导为6.75 GiB，口径分开。
- 两次候选及两次生产新256k，三事实均3/3；每次完整重复n_prefill=0；追问仅新增29token并回答7319。顺序包含长文→重复→追问→短文→新长文。追问引擎TTFT和HTTP总时长单独留在continuation JSON，不与新文档TTFT混用。
- 两次生产200-token NLL均2.165267，与参考2.164792之差+0.000475，PPL8.7169、top1=0.485。这些测例不代表通用长文质量或并发SLA。
- fnx.service active/enabled，health显示worker存活、两个新开关和实际快照字节数；回退bash /root/ascend-dev/r95_rollback.sh恢复r92。只回退快照策略也可在启动配置中设FNX_STATE_DUAL_SLIM=0后重启；旧格式仍兼容。
- r96仅做缓存写入算子探针，结果单独保存在benchmarks/r96，未据微基准直接替换生产实现。下一步应针对ScatterUpdate/评分融合和MoE/DeltaNet布局做单模块优化，再进行完整请求验收；MTP仍未启用，尚未达到硬件极限。
