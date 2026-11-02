# 第91–92轮：独立进程间ACL设备IPC与分大小传输

2026-11-02（Asia/Shanghai），192.168.18.206，2×Ascend 910B4，24+24层，256k。
源码起点 7e1fb600a6ceb295a0e3bd863dd5b3bd76ea42b9；同日重新测原host路径。

## 实现

使用CANN 9.1的 `ipc_mem_get_export_key` / `ipc_mem_set_import_pid` /
`ipc_mem_import_by_key(..., ENABLE_PEER_ACCESS)`；白名单只授权worker的bare TGID。
显式 `memcpy_async DEVICE_TO_DEVICE` 读取导入的设备指针，应用层不经host hidden缓冲。
第一阶段新增2×160MiB的ACL设备双缓冲，共320MiB；由ACL分配，不能只看torch allocated。
发布前等待生产者stream完成，复制完成后才发送pipe元数据；接收方复制并同步后计算。
槽位沿用已有两块在途上限，只有收到对应计算完成ack才复用。先关闭borrower，后释放owner。
数值计算、权重、BF16 KV、FP32状态不变。

环境开关 `FNX_PIPELINE_TRANSPORT`：
- `host`：原共享主存路径，不分配ACL peer缓冲。
- `hybrid`：T≥128的prefill走设备IPC，decode和不足128的尾段走原主存路径。
- `peer`：所有hidden走IPC，供诊断；小包更慢，本轮未作为完整模型生产选择。

当前生产选择 **hybrid**，已持久化到p275.sh / scripts/prod_pipeline.sh。
保留每卡独立进程；P2P不是通过合并进程或重新引入跨设备图捕获实现。

## 传输微基准

包含source准备拷贝、跨进程元数据/ack、destination拷贝和同步；每轮热身4次、计时20次，
两轮中位数取均值。mode顺序host/peer/host/peer，原始数组保留。
该表测单个有效载荷往返，不是推理端到端或流水重叠吞吐。

| tokens | 有效载荷MiB | host ms | peer ms | 降低 |
|---:|---:|---:|---:|---:|
| 1 | 0.0195 | 0.438 | 0.474 | -8.4% |
| 128 | 2.5000 | 0.877 | 0.575 | 34.4% |
| 512 | 10.0000 | 1.792 | 1.094 | 39.0% |
| 4096 | 80.0000 | 8.264 | 5.859 | 29.1% |
| 8192 | 160.0000 | 15.758 | 11.284 | 28.4% |

20KiB的小包额外IPC调用和同步抵消了拷贝收益，因此hybrid保持decode的原路径。
8192-token块为160MiB；完整256k每阶段边界约5GiB有效hidden，权重/KV不跨卡传输。
通信只占百秒prefill中的小部分，不能把约28%的大包通信改善当作28%的TTFT改善。

## 完整请求

服务器本机HTTP，模型已加载，新随机文档，262016输入+128实际输出；新prefill262012，
只有3-token模板命中。每次三位置检索均3/3；质量NLL2.165267，门限2.164792±0.01。

| 阶段 | TTFT s | Prefill s | TPOT ms | 输出tok/s | E2E s |
|---|---:|---:|---:|---:|---:|
| 同日host基线 1 | 102.114 | 100.844 | 54.04 | 18.50 | 108.977 |
| hybrid候选 1 | 101.890 | 100.626 | 54.29 | 18.42 | 108.786 |
| hybrid候选 2 | 101.454 | 100.235 | 54.31 | 18.41 | 108.351 |
| 生产hybrid 1 | 102.114 | 100.830 | 54.17 | 18.46 | 108.994 |
| 生产hybrid 2 | 101.607 | 100.365 | 54.16 | 18.46 | 108.485 |

生产三次新8064输入、128输出：TTFT中位 5.874s，
TPOT中位 52.68ms。
候选8k中位 5.873s，
同日host基线8k中位 5.879s。

收益小且样本有限；同日候选和生产样本分别列出，不声称统计显著，不宣称decode被P2P加速。
生产启动到就绪约201s；health新增pipeline_transport和peer/host的实际字节数与次数。
最终peer有效载荷计数 11214520320 bytes。

## 验收

- 1/7/128/512/4096/8192-token载荷，每档4次交替槽位、不同内容；任意BF16位模式按int16逐位比较，包括NaN位型。
- 非默认生产者stream的发布、两个同时在途槽位、正常borrower→owner关闭释放通过。
- 完整256k、原文重复n_prefill=0、继续追问262045输入且只新增29token，回答7319。
- 长文后连续三次新8k，以及短文后新256k通过；无新增OOM。
- 生产目录与Git对应模块AST一致；服务active/enabled。硬件拓扑为SYS、跨NUMA、无HCCS。

这里验证的是显式ACL设备IPC与D2D应用路径；没有使用历史copy_带宽来证明底层路由，
也没有用NPU DMA硬件trace量化物理链路占比。

## 复现与回退

在对应CANN/torch_npu环境，设置父进程可见设备0，子进程使用设备1：
`ASCEND_RT_VISIBLE_DEVICES=0 python tests/peer_ipc_probe.py`。
若要切回主存，在p275.sh将FNX_PIPELINE_TRANSPORT改为host并计划重启服务；不在请求中途自动切换。
完整回滚：`bash /root/ascend-dev/r92_rollback.sh`；备份 `/root/ascend-dev/r92/predeploy.tgz`。
原始证据 `benchmarks/r92/`，单卡进程图和双KV路径未改变。

API依据：
- [导出与白名单流程](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_1070.html)
- [导入与ENABLE_PEER_ACCESS](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/910/API/runtimeapi/aclpythondevg_01_1072.html)
