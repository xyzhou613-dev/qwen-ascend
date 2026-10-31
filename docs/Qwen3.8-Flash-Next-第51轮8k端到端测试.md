# 第51轮：8k总上下文端到端测试

日期：2026-10-31；机器192.168.18.206；r50生产配置，服务已加载模型。

输入8064 token，输出上限128 token，三轮均实际输出128 token，合计8192 token。
每轮prompt开头使用新的唯一编号；仅复用3个静态模板token，实际新prefill8060 token，
另一个prompt末token在生成首token步骤处理。没有对话前缀缓存命中。

从服务器本机HTTP客户端提交请求开始，到SSE `[DONE]` 收到为止计时。
首token计时忽略prefill进度注释及空delta。模型加载、payload准备及外部网络传输不计入。
当前电脑直连8000超时，SSH服务禁用TCP转发；没有修改防火墙或sshd配置。

| 轮次 | 输入 | 实际输出 | Prefill | 首token延迟 | 端到端 | Decode间隔 |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 8064 | 128 | 14.375 s | 14.643 s | 27.838 s | 103.90 ms |
| 2 | 8064 | 128 | 14.247 s | 14.430 s | 27.006 s | 99.02 ms |
| 3 | 8064 | 128 | 14.259 s | 14.444 s | 26.263 s | 93.06 ms |
| 中位数 | — | — | **14.259 s** | **14.444 s** | **27.006 s** | **99.02 ms** |

端到端范围26.263–27.838秒。端到端包含全部128-token生成，并非只统计prefill。
该结果是三轮近8k新prompt的测量，不包含模型冷启动，也不是远程客户端网络延迟测量。

原始结果：`/root/ascend-dev/r51/results.json`；日志：`/root/ascend-dev/logs/r51_e2e.log`。
脚本：`backend/remote/r51_payloads.py`和`backend/remote/r51_measure.py`。
本地结果副本：`_work/r51-results/results.json`、`_work/r51-results/r51_e2e.log`。
