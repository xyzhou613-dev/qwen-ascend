# 第54–55轮：256k低延迟部署

日期：2026-10-31。服务器192.168.18.206，2×Ascend 910B4，48层Qwen3.8-Flash-Next。
本轮确认目标：全新256k输入TTFT≤120秒，连续输出≥15 tokens/s，并保持长文检索正确。

## 已通过的完整256k测试

服务已加载，服务器本机HTTP；输入262016 tokens，输出上限64，实际输出44。
随机文档编号避免对话缓存命中，只有3-token静态模板前缀复用，新prefill为262012 tokens。
三个验收码分别放在正文约10%、50%、90%处。

| 指标 | r53已验证256k起点 | 本轮部署前验收 |
|---|---:|---:|
| Prefill | 607.678 s | 116.424 s |
| Prefill吞吐 | 431.17 tok/s | 2250.50 tok/s |
| 客户端TTFT | 624.785 s | **117.882 s** |
| 输出TPOT | 126.32 ms | **55.79 ms** |
| 连续输出速度 | 7.92 tok/s | **17.92 tok/s** |
| HTTP完整端到端 | 630.217 s | 120.281 s |
| 三事实检索 | 3/3 | **3/3** |

同一256k输入重复请求：新prefill为0，TTFT约84 ms（引擎计时）、HTTP端到端3.767秒，三事实再次全部正确。
TTFT包含本机HTTP、服务端tokenization、状态准备、prefill及首token；不含模型冷启动与外部网络传输。
以上为单次新prompt，不能解释为所有输入或并发负载的延迟保证。

## 最终实现

每张卡使用独立Python进程，各24层，每个进程只看到一个本地`npu:0`。
第二阶段通过继承的本机管道接收命令，用两个受限的文件映射缓冲区传递隐藏状态。
prefill可同时执行第一阶段的后一块和第二阶段的前一块；每阶段内部保持严格因果顺序。
请求计时结束前排空所有在途块，确保统计包含两卡完整计算。

输出采样在第二阶段NPU上完成，只返回token id；诊断接口仍可读取完整logits。
子进程跟随父进程退出；健康检查确认工作进程存活，工作进程退出时返回503。

4096-token主图，另捕获512和128-token图处理尾段，使用独立图内存池。
DeltaNet的64-token小系统批量准备，仅跨块状态更新保持顺序；递归状态仍为FP32。
随机及高相关key的65/512/2048/4096-token测试与原64-token求解路径输出及状态逐位一致。
2048-token求解微基准约27.62→8.03 ms。

KV改为按token连续的物理布局，暴露的逻辑形状仍为`[B,KVH,P,HD]`。
行更新和选中KV收集按第0维操作，配合BSND融合注意力，避免跨头重新物化KV。
KV保持BF16；布局前后更新的KV逐位一致，短注意力输出逐位一致。
融合选中注意力与原显式BF16权重注意力的平均相对差约0.2%–0.3%。
索引器缓存已池化、归一化并旋转的key，仅更新受影响的块。
查询、历史key变更及边界测试中，选中token位置与原路径逐位一致。

前缀快照仅复制有效位置；最近一份对话快照保留在各自NPU，减少长文状态回传主机的等待，并为最大窗口的临时工作区保留显存余量。
注意力投影预先反量化为BF16，保留原量化权重对应的值。
MoE仅启用无损scale就地乘和对角索引优化；仍用原FP32 router softmax及激活计算。

## 正确性门禁与舍弃的候选

最终基础200-token检查：NLL2.165267、PPL8.7169、top1=48.5%。
原起点NLL2.164792，差+0.000475，满足预设0.01门限。
32k及256k完整新输入和缓存复用均出现`LONG_ACCEPT`，三个验收码全部正确。
三事实检索属于本次输入的功能检查，不代表所有长文任务的准确率。

单进程下第二张卡整段或逐层图回放：短NLL和若干局部对拍可通过，但完整长文检索失败，未用于最终部署。
已经改用每卡独立进程，默认`FNX_GRAPH_G1=0 FNX_GRAPH_HEAD=0`。
MoE低精度router/激活候选NLL2.185833、FP32 decode BMM候选NLL2.177161，均超过质量门限，未启用。
最终`FNX_MOE_ROUTER_FAST=0 FNX_MOE_ACT_BF16=0 FNX_DN_DECODE_BMM=0`。

## 持久配置与回滚

默认`CTX=262144 LPC=24`，写入`/root/ascend-dev/r52/active_profile.env`。
GPU对话缓存最终设置为`FNX_CONV_CACHE=1 FNX_CACHE_GPU=1`。
`p275.sh`启用独立进程流水线、4096/512/128分块、FP32批量DeltaNet、索引缓存、BF16 KV布局与融合注意力。
`fnx.service`为active/enabled，健康检查返回ctx262144、pipeline_stages2、worker_alive=true。
部署后另做全新256k输入、128-token输出和三次8k请求，已全部通过下节门禁。

备份：`/root/ascend-dev/r55/predeploy.tgz`。
回滚命令（仍保留已验证的256k容量，恢复r53执行路径）：

```bash
bash /root/ascend-dev/r55_rollback.sh
```

主要日志：`/root/ascend-dev/logs/r55_layout256.log`、`r55_activate.log`、`r55_production256.log`、`r55_production8k.log`。
阶段结果：`/root/ascend-dev/r55/layout/result_262144.json`；生产结果：`r55/production/result_262144.json`。

## 部署后的独立验收

模型来源为生产目录`/root/ascend-dev/fnx/fnx`，通过systemd启动，使用新随机编号的输入。

| 项目 | 生产验收实测 |
|---|---:|
| 上下文上限 | 262144 tokens |
| 实际输入 / 输出 | 262016 / 128 tokens |
| 新prefill | 262012 tokens |
| Prefill | 116.610 s，2246.91 tok/s |
| 客户端TTFT | **117.861 s** |
| 输出TPOT / 速度 | **55.97 ms / 17.87 tok/s** |
| HTTP端到端 | 124.969 s |
| 三事实检索 | **3/3** |

`PRODUCTION_TARGET_ACCEPT`及`FINAL_PRODUCTION_ACCEPT`均通过。
相同输入重复请求：新prefill为0，HTTP端到端8.256秒，输出128 tokens，三事实仍正确。
三次全新256k请求的TTFT分别117.882秒（部署前44-token输出）、117.966秒（第一次生产验收128-token输出）、117.861秒（最终缓存配置128-token输出），均小于120秒，输出速度均超过15 tok/s。

保持默认256k窗口，另测三次新8k输入（8064 tokens）、每次输出128 tokens：

| 次数 | TTFT | HTTP端到端 | 输出TPOT |
|---|---:|---:|---:|
| 1 | 7.010 s | 14.118 s | 55.96 ms |
| 2 | 6.989 s | 14.097 s | 55.96 ms |
| 3 | 6.992 s | 14.105 s | 56.00 ms |

8k输入TTFT中位数6.992秒、端到端中位数14.105秒，三次均新prefill8060 tokens。
最终默认仍是256k，未为了8k测试切换到小窗口。

本地原始结果与日志：`D:\Ascend learning\_work\r55-results`。
