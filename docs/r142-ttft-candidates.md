# Flash Next r142（续）：两个 TTFT 候选的实测与否决

2026-11-11，192.168.18.206，2×Ascend 910B4-1。基于正式运行的 r141（a4eea95 / 1a36d30）。本轮先建立了 r141 的 decode/prefill 耗时账本（见 `r142-decode-budget.md`），随后针对账本指出的 TTFT 大头做了两个候选，**两个都没有部署**：一个未通过数值门槛，一个实测无收益。生产已恢复为 r141 原样。

## 0. 结论

| 候选 | 收益 | 结果 |
|---|---|---|
| B：首个 prefill 块 512→2048（4 次读权重变 1 次） | 8k TTFT **−10.7%**、256k −0.6% | **否决**：长提示输出重结合后分叉，无法证明质量不降 |
| C：索引打分按列分块（逐位安全） | 无 | 8k 逐字节一致；256k prefill 反而 **慢 1.1%**，否决 |

生产文件已逐字节恢复：`fnx/engine.py`、`fnx/pipeline.py` 与仓库一致，`p275.sh` 中本轮新增的两行已删除；8k 三份回答与 r141 逐字节相同，`model-switch.service` active、`RefuseManualStart=no`。

## 1. 候选 B：首块 512 → 2048

账本拟合显示 prefill 每个块有约 **125 ms/级** 的固定成本（每块都要重读全部稠密权重与全部 512 个专家的 W4），而 8k 请求的前 2048 token 被固定切成 **4×512**，等于把这个固定成本付了四遍。把首块放到 2048（仍在 pos+T≤2048 的 dense/short 区间内）即可只付一遍。

实现：`fnx/pipeline.py` 增加 `FNX_PIPELINE_FIRST_BLOCK`（0=512，1=2048，默认 0），`p275.sh` 参数化透传；补丁脚本 `ops/r142-first-block.py`（先 `--check` 再写入，带 bash 语法检查）。

实测（同机 HTTP，8k 为 3 轮中位数，256k 单次）：

| 指标 | r141 基线 | 候选 B |
|---|---:|---:|
| 8k TTFT | 4.335 s | **3.871 s（−10.7%）** |
| 8k prefill | 4.251 s | 3.787 s（−10.9%） |
| 8k TPOT | 34.77 ms | 34.61 ms（decode 未改，噪声内） |
| 256k TTFT | 73.653 s | 73.228 s（−0.6%） |
| 256k TPOT | 35.99 ms | 35.89 ms |

数值检查：

- 三份 8k 完整回答与 r141 **不再逐字节一致**，且分叉点很早（第 20–23 个字符后即不同）。这是分块重结合（dense attention 的 online-softmax 分块、MoE 分组、DeltaNet 分块批尺寸都变了）导致的预期行为，不是可以忽略的末位差异。
- 生产 `/fnx/loss` 的 200-token NLL 仍是 2.165267，但**该口径对本改动没有判别力**：`/fnx/loss` 是逐 token `step_pad` 的 decode 路径，根本不走分块 prefill（见 §3）。
- 为补这个缺口新增了 `prefill_nll.py`：用真实 8064-token payload 走生产分块策略 prefill 7864 token，再对最后 200 个位置做 teacher-forced NLL。结果：基线 **14.908**、候选 **15.766**（top1 都是 4.02%）。该文本 NLL≈15 属于模型几乎在猜的混沌区，只能说明"两者预测确有差异"，不能用来判优劣。

结论：这是改变 FP32 累加/归约路径的改动，按项目纪律（不得"改变递推求和顺序后只检查 argmax"）**不部署**。补丁保留在 `ops/r142-first-block.py`，默认关闭；下一步需要一份自然长文（NLL 2–4 区间）的质量口径才能裁决。

## 2. 候选 C：索引打分按列分块

账本显示长文晚段块里索引链占 17%，其中打分矩阵 `[4×512, 65536]` 每次要整块写回再读（约 1.2 GB/层·tile）。按列切块（`FNX_INDEX_SCORE_TILE=4096`，`score` 原生核要求列数是 2048 的倍数）可以让中间量留在 cache，且**逐列点乘与逐列 relu/sum 都不变**，属逐位安全改动。补丁：`ops/r143-score-tile.py`。

| 指标 | r141 基线 | 候选 C |
|---|---:|---:|
| 8k TTFT | 4.335 s | 4.332 s |
| 8k TPOT | 34.77 ms | 34.43 ms |
| 8k 三份回答 | — | **逐字节一致** ✓ |
| 256k prefill | 72.408 s | 72.777 s |
| 256k TTFT | 73.653 s | 74.051 s |

256k 侧与 r141 验收口径比是 +0.5%，而同轮基线自身的重复波动就有 0.7%（r141 验收时 baseline256 两次为 73.095/73.637），所以准确说法是**没有可分辨收益**：列切块把 192 次打分拆成 1536 次，多出的发射与 `keys[:, c0:c1]` 非连续切片拷贝抵消了 cache 收益。否决，未部署（默认 0）。

## 3. 口径缺口：`/fnx/loss` 不能验证 prefill 改动

`POST /fnx/loss` 逐 token 调 `step_pad`，走的是 decode 路径，因此**任何 prefill 分块/索引改动都不会被它检出**。历史轮次（r125 中等分块、r131 分块策略）引用 NLL 2.165267 作为精度证据，实际只覆盖了 decode 路径。

本轮新增的 `prefill_nll.py` 补上了这条路径：`plan_prefill` + `prefill_chunk` 走生产分块，再对末 200 个位置 teacher-forced 打分。它的输出对分块变化敏感（本次基线 14.908 vs 候选 15.766），但要成为质量门槛还需要一段**自然、非重复、NLL 在 2–4** 的长文；当前 8k 基准 payload 是重复填充文本，末段是问题模板，NLL≈15，判别力不足。

## 4. 证据与复现

- 远程：`/models/fnx-diagnostics/r142-budget-20261111/`（`baseline8k`、`candidate8k`、`candidate256`、`tile8k`、`tile256`、`nll8k-*.log`、`p275-*.log`、`ship-health.json`）。
- 本地：`r142-ttft-evidence/`（results.json、NLL 日志、脚本、sha256）。
- 脚本：`ops/r142-first-block.py`、`ops/r143-score-tile.py`（都可 `--check` 后应用，默认值不改变生产行为）。
- 采集依赖：`launch_nll.py` + `prefill_nll.py` 需要诊断目录里的 `fnx` 副本与生产环境变量快照，不是独立程序。

## 5. 服务状态

恢复后 `/health`：`service_revision=r141`、`ctx=262144`、`worker_alive=true`；8k 三份回答 sha256 与 r141 验收完全相同；`model-switch.service` active、`RefuseManualStart=no`，本轮三个临时 drop-in 均已删除。生产 `engine.py`、`pipeline.py`、`p275.sh` 与 r141 逐字节一致。
