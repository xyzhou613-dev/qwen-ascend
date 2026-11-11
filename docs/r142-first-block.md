# Flash Next r142：首块 2048 上线（TTFT −10.8%）与索引列分块否决

验收完成：2026-11-11T18:48+08:00，正式版本 **r142**。基线 r141（a4eea95 / 1a36d30）。
设备 192.168.18.206，2×Ascend 910B4-1（64 GiB/卡）；CANN 9.1.0、驱动 26.1.1、torch 2.10.0 / torch_npu 2.10.0.post4；262144 窗口、9002 端口。

## 0. 结果

| 指标（同机 HTTP，8k=8064 输入×128 输出×3 轮中位数） | r141 | r142 | 变化 |
|---|---:|---:|---:|
| 8k TTFT | 4.335 s | **3.866 s** | **−10.82%** |
| 8k prefill | 4.251 s | 3.782 s | −11.03% |
| 8k HTTP 端到端 | 8.751 s | 8.282 s | −5.36% |
| 8k decode TPOT | 34.77 ms | 34.77 ms | 0（decode 未改） |
| 256k TTFT（各一次） | 73.653 s | 73.215 s | −0.59% |
| 256k prefill | 72.408 s | 71.902 s | −0.70% |
| 256k TPOT | 35.99 ms | 35.90 ms | −0.25% |
| 256k 三事实召回 | 3/3 | 3/3 | — |

8k 三轮（r142 正式版本复测）：TTFT 3.966 / 3.866 / 3.863 s，prefill 3.858 / 3.782 / 3.781 s，TPOT 34.75 / 34.77 / 34.78 ms；端到端 8.379 / 8.282 / 8.280 s。
256k 一次：n_prefill=262012（未命中缓存），重复请求缓存与三事实召回均通过。

## 1. 改动

`fnx/pipeline.py`：短块（dense attention，`pos+T<=2048`）区间的**首个 prefill 块**由固定 512 行改为可配置，生产取 **2048** 行；`p275.sh` 以 `FNX_PIPELINE_FIRST_BLOCK`（默认 1）持久启用。回退：置 0 后重启。

动因来自 r142 账本：prefill 每个块有约 **125 ms/级** 的固定成本（每块都要重读全部稠密权重和全部 512 个专家的 W4）。8k 请求的前 2048 token 原来被切成 **4×512**，等于把这份固定成本付了四遍；合成一块后 8k prefill 少约 0.47 s/级。

不改的东西：权重、量化、KV/FP32 递推格式、decode 路径、事件与算子实现、256k 窗口与模型切换。

## 2. 数值边界（本轮的核心工作）

这项改动**会改变输出**：它改变 dense attention 的 online-softmax 分块、MoE 分组规模与 DeltaNet 分块批尺寸，属于重新结合，不是逐位不变的实现替换。所以先做了质量门槛，通过后才发布：

| 口径 | r141 | r142 | 说明 |
|---|---:|---:|---|
| 自然长文 6000 token，末 200 token teacher-forced NLL | **1.745571** | **1.722880** | 用生产分块策略真实 prefill 后打分；候选更优 |
| 同上 top1 | 59.30% | **61.31%** | 候选更优 |
| 200-token `/fnx/loss` NLL | 2.165267 | 2.165267 | 该口径是逐 token decode，**不经过分块 prefill**，对本改动无判别力，只用于防止 decode 侧回归 |
| 256k 三事实召回 | 3/3 | 3/3 | 长文事实保持 |

三份 8k 完整回答与 r141 **不再逐字节一致**（这是预期：分块重结合后首个近并列 token 会翻转）。r142 起以新的三份回答为参考：
`f711cb62…`、`b0ffc39e…`、`0dd57e9f…`（旧 r141 为 `16268946…`、`b600d070…`、`713f0a0d…`）。两次独立部署（候选与正式）得到的三份回答 sha256 相同，可复现。

同时修正了一处长期口径问题：`POST /fnx/loss` 逐 token `step_pad`，走的是 decode 路径，**任何 prefill 分块改动都不会被它检出**；此前用它给分块类改动背书是无效的。本轮新增 `prefill_nll.py`：走生产 `plan_prefill`/`prefill_chunk`，再对末 200 token 做 teacher-forced NLL，才第一次覆盖到被改的路径。

## 3. 被否决的候选：索引打分按列分块

账本显示长文晚段块里索引链占 17%，其中打分矩阵 `[4×512, 65536]` 每次要整块写回再读。按列切块（`FNX_INDEX_SCORE_TILE=4096`，`score` 原生核要求列数为 2048 的倍数）逐列点乘与 relu/sum 都不变，属逐位安全改动：

| 指标 | r141 | 列分块 |
|---|---:|---:|
| 8k 三份回答 | — | 逐字节一致 ✓ |
| 8k TTFT | 4.335 s | 4.332 s |
| 256k prefill | 72.408 s | 72.777 s |

没有可分辨收益（192 次打分拆成 1536 次，多出的发射与 `keys[:, c0:c1]` 非连续切片拷贝抵消了 cache 收益），未部署。补丁保留在 `ops/r143-score-tile.py`（默认关闭）。

## 4. 部署、回退与证据

- 部署：`ops/r142-first-block.py /root/ascend-dev`（先 `--check`，带 bash 语法检查）→ `p275.sh` 里 `FNX_PIPELINE_FIRST_BLOCK` 默认 1 → 通过既有入口重启（`p275.sh` 或 `modelctl.sh`）。
- 回退：`FNX_PIPELINE_FIRST_BLOCK=0` 重启即回到 r141 的分块方式；完整文件回退把 `fnx/pipeline.py`、`fnx/server.py` 还原到 r141 提交即可。
- 健康检查：`service_revision=r142`、`ctx=262144`、`worker_alive=true`、`short_prefill_chunk=2048`；`model-switch.service` active、`RefuseManualStart=no`。
- 证据：`benchmarks/r142-ttft/`（results、256k 结果、NLL 日志、脚本）；远程 `/models/fnx-diagnostics/r142-budget-20261111/`；本地 `r142-ttft-evidence/`。
- 仅验证当前软硬件矩阵；其他 910B 环境需自检、重跑分块 NLL 门槛并重新验收。

## 5. 下一步（已核实的方向）

`/models/Qwen3.8-Flash-Next` 的 index 里有 31 个 `mtp.*` 张量（`fc_embedding`/`fc_hidden`/`pre_fc_norm_*` + `mtp.layers.0` 的 attention、indexer、MoE 专家），`config.json` 的 `mtp.num_hidden_layers=1`。若以 MTP 头做草稿、对目标模型做精确贪心校验，被接受 token 与不投机时逐 token 相同，是当前唯一可能拿到 1.5–2× decode 而不牺牲数值的方向；工作量在 MTP 前向、草稿循环与拒绝回滚。
