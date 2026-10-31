# 第49轮：分块图 prefill 与 QSA 长上下文

日期：2026-10-31。机器：192.168.18.206，2 × 910B4。

## 现网基线

运行版本为远端仓库 `66d7bd4`（r48.3），并非本地第46轮文档中的快照。
`FNX_CTX=2048`、`FNX_LPC=45`、48层、`FNX_DEQ_ATTN=0`。
历史服务验收 NLL=2.1648、PPL=8.713、top1=48%；decode 63.4–63.7 ms/token。

## 改动

1. 新增 dev0 的固定块 prefill NPUGraph，与 decode 共用图内存池；两图串行重放。
   设备端位置张量参与 RoPE、KV 更新及 causal mask，防止捕获时将起始位置固化。
   dev1 保持 eager；块尾、缓存未对齐部分仍可走原路径。
2. 新增 QSA indexer：4个 query head，128维；每4个 raw key 做均值池化，
   RMSNorm + RoPE 后按 `sum(relu(QK))/sqrt(128)` 选512个完整块，并保留当前不完整尾块。
   未完成的未来块先屏蔽；分块中每个 query 有自己的 causal mask。
3. indexer 的 raw key cache 加入 reset、save_state、load_state、save_into、图自检快照，
   避免不同请求及前缀缓存之间串状态。
4. 请求入口检查 prompt + max_tokens 的容量，防止生成期间 KV 越界。
5. `/health` 展示 chunk、图和 indexer 开关；响应增加实际 prefill 时间与吞吐，
   删除基于旧143 ms/token常数的等待估计。

默认开关保持关闭；启用使用 `FNX_CHUNK=8 FNX_CHUNK_GRAPH=1 FNX_INDEXER=1`。
indexer 和分块目前限定 `FNX_BATCH=1`。

## 已完成的验证

- `t_prefill_audit.py`：8个 token，逐 token eager vs chunk=1/8，状态及 logits 逐位相等。
- `t_chunk_graph_accept.py`，64个真实文本 token：16.50 → 34.21 tok/s（2.073×）；
  logits maxdiff=0、NLL差=0、top1全一致，chunk执行后decode仍逐位不变。
- `t_indexer_accept.py`：位置0、3、2047、2048、2051、4095、8191，选块mask与逐query参考写法完全一致；
  图重放位置2047→2051→4095，mask逐位一致。2048位置允许保留2048个完整块token外加1个尾token，
  这与参考实现语义一致。

- `t_chunk_graph_accept.py`，P=8192、indexer开启、2112个真实文本 token：
  逐token 182.836 s（11.551 tok/s），chunk=8 为22.925 tok/s（1.985×）；
  logits maxdiff=0、NLL差=0、top1全一致。该次未额外重复完整2112-token基准，
  decode图共享池的往返检查由上述64-token测试覆盖。
- 线上8k服务短文本质量：NLL=2.164792、PPL=8.7128、top1=48%、n=200。

生产已设置 `FNX_CTX=8192 FNX_CHUNK=8 FNX_CHUNK_GRAPH=1 FNX_INDEXER=1`。
`/health` 显示ctx=8192、chunk_graph=true、qsa_indexer=true；systemd为active/enabled。
线上真实长请求：7430 prompt token，prefill=330.194 s（22.49 tok/s），
总耗时331.229 s；回答末尾的3+5题为“3加5等于8。”。
重复请求命中7429-token状态缓存，prefill=0、总耗时0.766 s（客户端0.798 s），回答一致。
超出prompt+输出预算的请求返回400。该次长请求在下述短图优化之前测量。

## 短窗口图优化

8k通用图虽然正确，短输出TPOT曾增加至94.64 ms。原因是每步都计算8k窗口和稀疏选块。
新增与长图共享内存池的2k短decode/prefill图；在宿主记录已处理位置，按块末位置选择图。
2k以内无需选块，但仍记录indexer raw keys，保障后续跨过2k边界和缓存续接。
`load_state/reset/step/prefill_chunk` 同时维护宿主位置。

P=8192下短图验收：64个真实token，逐token16.208 → 分块34.295 tok/s（2.116×）；
logits、NLL、共享池decode往返均逐位一致。额外从位置2040开始运行64 token，
跨过2048时短/长图切换与分块结果maxdiff=0。

同时纠正非流式接口TPOT统计：此前将全部prefill时间除以输出token数，
导致该长请求的`tpot_ms`虚报47318 ms；现在分别记录TTFT、真实prefill时间和输出token间隔。

最终服务短请求复测（96个输出token）：TPOT=68.35 ms/token，已消除8k通用路径曾出现的94.64 ms回退；
相对原2k基线63–64 ms仍有小幅开销，主要包括维护长上下文indexer缓存。
最终NLL仍为2.164792，PPL=8.7128、top1=48%。

最终服务跨2k验收：2130 prompt token，prefill=63.028 s（33.73 tok/s），
TTFT=64.575 s，总耗时65.119 s，回答“3加5等于8。”；
缓存复用后TTFT=113.90 ms、总耗时0.652 s，回答一致。
长位置decode约90 ms/token，短位置约68 ms/token；长窗口仍有明显的注意力/选块成本。
超预算请求400，`ONLINE_ACCEPT`通过。该结果来自最终部署版本。

最终运行文件MD5（LF换行，格式归一化前后AST一致）：

```text
engine.py df4d7910474f6d40e213885208b2dfc6
model.py  828b41d0ffbb40161048c68678eb8981
server.py 1d5ffee5259885b1d04fbacfc31e6363
```

运行代码已同步至 `/root/qwen-ascend` 工作区，未推送远端；记录在其 `docs` 目录。
原始验收日志保留在 `/root/ascend-dev/logs/r49_*.log`。

## 运维与回滚

生产改动前备份位于 `/root/ascend-dev/r49/baseline.tgz`，包含 `fnx/fnx`、`p275.sh`、`watchdog.sh`；
原环境位于 `/root/ascend-dev/r49/baseline.env`。实验在 `/root/ascend-dev/r49/fnx` 隔离副本运行。
停止服务做独占卡测试时，shell EXIT trap 会重新启动 systemd 服务。

回滚需先停止服务及容器内服务进程，再将备份解回 `/root/ascend-dev`，最后启动 `fnx.service`。
不能只执行 `systemctl stop` 就认为卡已经释放，容器内 `docker exec` 的 Python 需显式退出。

## 范围

这是保持现有量化及禁用PLE策略的后端优化；并不表示与原始全精度模型全网逐位等价。
长上下文首先验证8k；更大窗口仍需重新检查图工作区、KV、缓存快照及延迟。
当前仍逐token执行 DeltaNet 递推及专家计算，后续进一步提速应测量 grouped MoE 和融合递推。
