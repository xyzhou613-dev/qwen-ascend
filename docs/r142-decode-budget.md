# Flash Next r142：r141 decode/prefill 耗时账本与两项候选否决

采样日期：2026-11-11。设备 192.168.18.206，2×Ascend 910B4-1（64 GiB/卡）；CANN 9.1.0、驱动 26.1.1、torch 2.10.0 / torch_npu 2.10.0.post4；262144 窗口、9002 端口。样本为**正式运行的 r141**（实现 a4eea95，验收 1a36d30）。

本轮没有改动生产代码：两项候选分别在实现前和实测后被否决，正式部署仍为 r141，8k TPOT 34.75 ms、256k TPOT 35.99 ms 不变。本轮交付的是新账本与否决证据。

## 1. 方法

独占采样：暂停 `model-switch.service`、写 `RefuseManualStart=yes` 临时 drop-in、停正式引擎，再把 r141 运行包整份复制到 `/models/fnx-diagnostics/r142-budget-20261111/fnx`，只给 `pipeline_worker.py` 加 `prof_start/prof_stop/inventory` 三个 RPC。其余源码、共享库、环境变量与生产一致。

`torch_npu.profiler`，CPU+NPU 活动，`profiler_level=Level1`，`AiCoreNone`（timing pass）。decode 采样 32 步，按**相邻步结束时刻**切出 31 个 token 周期做账；prefill 采一个块区间。账本闭合口径：

```text
token 周期 = 设备忙 + 级内空隙 + 级间/宿侧空隙
```

同一进程里先跑不带 profiler 的基线，再开 profiler 采样；两者不能混用。

## 2. 结论：r141 decode 的可加时间账本

无 profiler 基线（本进程，39 个周期）：8k **34.262 ms**（中位 34.224）、256k **35.474 ms**（中位 35.456）。带 profiler 的采样周期为 8k 34.686、256k 35.950，比基线慢 1.2%/1.3%，因此账本只用于归因，不作为目标数字。

| 每 token | 8k, ms | 256k, ms |
|---|---:|---:|
| 设备忙（kernel 并集） | 32.607 | 33.821 |
| 级内任务间空隙 | 1.135 | 1.154 |
| 级间 / 宿侧空隙 | 0.944 | 0.975 |
| **合计** | **34.686** | **35.950** |

两卡在 decode 中**不重叠**：stage0 与 stage1 串行，每 token 设备时间是两级之和（8k 两级设备忙分别约 15.9 / 16.7 ms）。因此 TPOT 只能靠减少单级工作，不能靠"两卡并行"。

### 2.1 分类（每 token）

| 类别 | 8k 调用 | 8k ms | 256k 调用 | 256k ms |
|---|---:|---:|---:|---:|
| 稠密 + 索引 MatMul | 530 | 11.897 | 530 | 12.378 |
| W4/W8 量化矩阵乘 | 112 | 6.435 | 112 | 6.405 |
| 其它向量/标量算子 | 2117 | 5.290 | 2117 | 6.103 |
| 原生融合算子 | 542 | 4.348 | 542 | 4.290 |
| FP32 DeltaNet | 36 | 1.283 | 36 | 1.275 |
| Gather/布局/缓存写 | 363 | 1.181 | 363 | 1.205 |
| RMSNorm | 181 | 1.119 | 181 | 1.108 |
| 稀疏选中注意力 | 12 | 0.448 | 12 | 0.449 |
| 卷积 | 36 | 0.307 | 36 | 0.307 |
| MoE 输出融合（r141） | 48 | 0.272 | 48 | 0.268 |

256k 比 8k 慢约 1.26 ms，其中向量类 +0.81 ms、MatMul +0.48 ms，来自长上下文索引打分/选择的增长，不是 attention 本体（0.45 ms 两档相同）。

### 2.2 逐投影（8k，每 token）

| 投影 | 调用 | ms | 单次 µs | 有效带宽 |
|---|---:|---:|---:|---:|
| w_qkv [10240,2560] | 36 | 2.308 | 64.1 | 819 GB/s |
| w_z [6144,2560] | 36 | 2.124 | 59.0 | 534 GB/s |
| w_out [2560,6144] | 48 | 1.841 | 38.4 | 828 GB/s |
| a_up+m_up [10240,320] | 96 | 1.515 | 15.8 | 417 GB/s |
| a_di+m_di [324,10240] | 96 | 1.234 | 12.9 | 512 GB/s |
| w_q [12288,2560] | 12 | 0.960 | 80.0 | 786 GB/s |
| sh_gu [1793,2560] | 48 | 0.752 | 15.7 | 585 GB/s |
| w_a/w_b [48,2560] | 72 | 0.468 | 6.5 | 31 GB/s |
| sh_down [2560,640] | 48 | 0.384 | 8.0 | 821 GB/s |
| w_k/w_v [512,2560] | 24 | 0.180 | 7.5 | 350 GB/s |
| exp_dn [10,640]×[640,25600] | 48 | 2.416 | 50.3 | 164 GB/s |
| exp_gu [1,2560]×[2560,12800] | 48 | 2.021 | 42.1 | 390 GB/s |
| lm_head [1,2560]×[2560,62080] | 4 | 1.685 | 421.3 | 378 GB/s |
| indexer W8 [1,2560]×[2560,640] | 12 | 0.317 | 26.4 | — |

`lm_head` 的 4 次调用是同一个 62080 列 GEMM 的内部分块（每次 421 µs、形状相同、串行），不是重复计算；r139 已测过 bf16 常驻 1.00 ms 更快但末位不同，未采用。

### 2.3 单算子（8k，每 token）

MatMulCommon 530 次 11.900 ms、WeightQuantBatchMatmul 112 次 6.439 ms、gather_pair 96 次 2.468 ms、delta_sequential 36 次 1.283 ms、RmsNorm 181 次 1.120 ms、Mul 338 次 0.959 ms、Cast 605 次 0.952 ms、ReduceSum 132 次 0.862 ms、post_mul 133 次 0.694 ms、residual_inject 96 次 0.660 ms、FusedInferAttentionScore 12 次 0.448 ms、GatherV3 72 次 0.403 ms、hc_sigmoid_reduce 97 次 0.329 ms、conv_four 36 次 0.307 ms、Softmax 48 次 0.285 ms、moe_cartesian_epilogue 48 次 0.272 ms、ConcatD 96 次 0.266 ms、Topk 60 次 0.257 ms。

每 token 设备 kernel 约 4100 次。原生融合类（542 次 / 4.35 ms）与"其它向量"（2117 次 / 5.29 ms）合计 9.6 ms，占设备时间 29%，平均每次 2.5–8 µs，属发射/依赖开销，是 r140/r141 路线的剩余空间。

## 3. prefill 账本

口径与 decode 不同：一次只采一个块区间，窗口内设备时间不可再按 token 归一。

| 块 | 位置 | 设备 ms | 空闲 ms | 说明 |
|---|---:|---:|---:|---|
| 512 token（首块） | 0 | 490.1 | 129.3 | 两级忙 248.8 / 241.3 |
| 8192 token（末段） | 247808 | 4527.6 | 145.1 | 两级忙 2255.5 / 2272.1 |

8192 行末段块的设备时间构成：

| 类别 | 调用 | ms | 占比 |
|---|---:|---:|---:|
| 其它向量/标量 | 31337 | 1732.7 | 38% |
| 稠密 + 索引 MatMul | 25908 | 886.2 | 20% |
| 稀疏选中注意力 | 24 | 684.2 | 15% |
| MoE 分组 GMM | 96 | 610.7 | 13% |
| Gather/布局/缓存写 | 13538 | 417.8 | 9% |
| RMSNorm | 180 | 93.3 | 2% |
| HC/MoE 融合归约 | 144 | 82.6 | 2% |
| 卷积 | 36 | 39.9 | 1% |

单算子：SparseFlashAttention 24 次 684.2 ms、GroupedMatmulV5 96 次 610.7 ms、BatchMatMulNd 20448 次 475.9 ms、MatMulV3 468 次 333.6 ms、TopkV2 240 次 301.9 ms、Mul 6228 次 250.6 ms、Add 10344 次 201.9 ms、Cast 2184 次 191.4 ms、LessEqual 192 次 178.1 ms、score_groups 192 次 134.4 ms、Tril 144 次 100.6 ms、MaskedFill 480 次 72.6 ms。

读法：

- 首块（512 token）里 MoE GMM 占 58%——该块激活了全部 512 个专家但每专家只有约 10 行，是**权重带宽**受限。
- 末段 8192 行块里 MoE GMM 降到 13%，DeltaNet 分块递推（BatchMatMulNd + 向量链）成为第一大类，约 2.6 s（58%）；索引选择链（Topk + LessEqual + Tril + MaskedFill + score_groups）约 0.79 s（17%）；稀疏注意力 0.68 s（15%）。
- 因此 TTFT 的下一步不在 decode 的那些小投影，而在 DeltaNet 分块递推与长文索引选择。r141 的 delta 分块初筛只测了组件，未覆盖这两条链在 8192 行块里的占比。

## 4. 本轮候选与否决理由

| 候选 | 预期 | 结果 |
|---|---|---|
| 融合 DeltaNet q/k 的 l2norm 链（5 算子×72/ token，约 1.25 ms） | −0.9 ms | **实现前否决**：手写 fp32 求和无法逐位复现 torch ReduceSum |
| gather_pair 双缓冲（2×40 KiB UB，MTE2/MTE3 重叠） | −0.7 ms | **实测否决**：组件 1.10×/1.00×，真实层无收益 |

### 4.1 l2norm 融合

`(x*x).sum(-1)` 的 fp32 归约顺序必须一致，否则 `rsqrt` 后的每个元素都会变。用 64×16×128 的随机张量对比 torch 的 `(x*x).sum(-1)` 与 8 种手工结合顺序（顺序累加、2/4/8/16/32/64 路、成对树、8 宽块+树），逐位一致率最高只有 **72.9%**（成对树），顺序累加仅 25.8%。结论：不引入该内核。

### 4.2 gather_pair 双缓冲

原内核每个 40 KiB 瓦片都是"加载→等→写出→等"完全串行。改成两块 40 KiB 暂存后，组件 A/B（真实形状、10 个 id）：

| 形状 | 原 | 双缓冲 | 加速 |
|---|---:|---:|---:|
| gu [E,2560,160] | 13.75 µs | 12.46 µs | 1.10× |
| dn [E,640,320] | 7.09 µs | 7.10 µs | 1.00× |

真实 L00 decode（图内 8 次 replay×5 组取中位）：原 1.91346 ms vs 双缓冲 1.91374 ms，**无收益**；输出、FP32 递推状态、BF16 卷积历史逐位一致。未部署，生产 `.so` 未改动。

### 4.3 附带发现（未部署）

生产 `libfnx_expert_gather.so`（sha256 `b7c4eaef…`）**不能由仓库源码复现**：用 `kernels/expert_gather/gather.cpp` 重新编译得到 `5e9457b1…`，双缓冲版为 `fc68d351…`。在独立测试里，重编版与双缓冲版对 torch 参考实现 3/3 逐位一致且确定性，生产二进制在 gu 形状（K=2560/G=40/N=1280）上不确定且与参考不符；但在真实 L00 decode 图内三者输出逐位相同，已验证路径不受影响。该差异未部署，留作后续核查（重新构建/替换该 `.so` 前必须先解释这一差异）。

### 4.4 2026-11-12 r143补充复核

通过生产wrapper、在输入准备同步就绪后，对生产gather库检查两种实际形状、边界/重复ID，共18次，均与Torch参考逐位一致且重复可复现。本轮未复现旧独立probe中的不确定性。旧probe缓存stream后又异步zero_等调用时序需先排除；不同构建路径产生的ELF SHA差异本身也不能证明源码语义不一致。本轮未替换生产gather库。此复核不等于已经完成可复现构建追踪，详情和源测试见[r143记录](r143-index-post.md)及benchmarks/r143-index-post/gather_check.py。

## 5. 证据与复现

- 远程原始目录：`/models/fnx-diagnostics/r142-budget-20261111/`（`traces/`、`traces2/` 为原始 profiler 文本导出，约 0.6 GB，不入库）。
- 本目录：`summary.json`（机器可读结论）、`time_budget.json`、`runs.json`、`runs2.json`、`analyze.py`、`profile_model.py`、`profile_prefill.py`、`profile_support.py`、`probe_sum_order.py`、`probe_gather.py`、`probe_gather2.py`、`layer_probe.py`、`gather_db.cpp`。
- 采集脚本不是独立程序：`profile_*.py` 依赖诊断目录里的 `fnx` 副本与 `pipeline_worker` 的 profiler RPC；不要覆盖生产服务。

## 6. 服务状态

采样结束后按原环境重启正式引擎，`/health` 返回 `service_revision=r141`、`ctx=262144`、`worker_alive=true`；`model-switch.service` 为 `active`、`RefuseManualStart=no`，临时 drop-in 已删除。生产代码、权重、共享库与启动参数均未改动。
