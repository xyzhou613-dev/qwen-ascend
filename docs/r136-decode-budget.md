# Flash Next r135：43.37 ms/token 算子与访存账本
> 采样日期：2026-11-10；设备：192.168.18.206，2 × Ascend 910B4-1（每卡 64 GiB）；本次分析编号 r136。

## 1. 结论与适用范围

**同轮完整时间线闭合为 43.3514 ms/token（23.0673 tokens/s）。** 原先 23.06 tokens/s 的倒数为 43.3651 ms，约为 43.37 ms；差值约 0.019 ms。没有缩放实测数据来强行凑数。

范围是单请求、batch=1、8064 token 输入、约 8k 位置的稳态 decode。模型仍配置 262144 上下文窗口；本报告不是 256k 已填满上下文时的 decode 账本，也不是 prefill 账本。

三次未启用 profiler 的 HTTP 测试 TPOT 分别为 43.33、43.33、43.24 ms；原路径独立 generate 基线均值为 42.8909 ms。时间线、访存 PMU、HBM 控制器和校准分别采集，不能把不同轮的时长重复相加。

## 2. 可相加的实测时间账本

| 项目 | ms/token | 占比 | 观测含义 |
|---|---:|---:|---|
| Kernel 执行并集 | 41.204574 | 95.05% | 两卡 kernel 的时间区间并集，已包含内部计算、访存、启动等 |
| 显式设备拷贝（扣除与 kernel 重叠） | 0.029169 | 0.07% | 来自 device task 时间线 |
| 阶段内部任务间空隙 | 1.175669 | 2.71% | 每阶段首尾任务之间扣除设备活动 |
| 宿侧 / 跨阶段 / token 边界空隙 | 0.942023 | 2.17% | 每 token 周期扣除两阶段任务跨度 |
| **合计** | **43.351436** | **100%** | 31 个完整 token 周期 |

```text
43.351436 = 41.204574 + 0.029169 + 1.175669 + 0.942023 ms
```

采样捕获 step 8～39，使用 end(step8)→end(step39) 的 31 个完整周期；排除 profiler 启停边界。两卡 kernel 重叠量在浮点舍入误差范围内为零；当前单流 PP24+24 decode 基本串行。stage0 kernel 忙时 19.7003 ms，stage1 21.5043 ms。

**“阶段内空隙”不能全部归因为硬件依赖等待；它也可能来自调度、提交和同步。“宿侧空隙”是时间线分类，不能全部称为 CPU 计算。kernel 内部流水线停顿已经算在 kernel 时间内。**

## 3. 各类算子的真实 kernel 时间和接口访存计数

下表时间来自低扰动 timing pass；字节来自另一轮 MemoryAccess pass。GB=10⁹ B。访存列为 AI Core/Vector 的 main-memory 接口 PMU 计数，**不是逐算子 HBM 控制器读写量**。

| 类别 | 调用/token | kernel ms/token | 接口读 GB/token | 接口写 GB/token |
|---|---:|---:|---:|---:|
| 普通投影 / 索引 MatMul | 530 | 14.0975 | 7.3610 | 0.0056 |
| 其他向量 / 标量算子 | 3476 | 8.7268 | 0.1421 | 0.0653 |
| Gather / 布局 / 缓存写入 | 651 | 6.5585 | 2.1983 | 2.1556 |
| MoE down GMM | 48 | 4.2658 | 2.0984 | 1.5947 |
| W4/W8 量化 MatMul | 64 | 4.0729 | 3.3367 | 1.6556 |
| FP32 DeltaNet | 36 | 1.3306 | 0.1190 | 0.1143 |
| RMSNorm | 181 | 1.1188 | 0.0257 | 0.0056 |
| HC / MoE 融合归约 | 145 | 0.5840 | 0.0117 | 0.0011 |
| 稀疏注意力 | 12 | 0.4497 | 0.0594 | 0.0076 |
| **合计** | **5143** | **41.2046** | **15.3524** | **5.6056** |

MemoryAccess pass 本身为 45.0052 ms/token，存在采集扰动，因此不使用其耗时替代 timing pass。原始 kernel 类型、调用次数、时间、读写量见 `kernel_types.csv`。

## 4. 主要数学计算量：逐投影核算

FMA=2 FLOPs；每个矩阵按实际逻辑 N、K 和被选中的 10 个专家计算。下面是模型有效数学量，不含硬件分块补齐、反量化、冗余指令；kernel 时间则包含这些实现成本。

| 投影 / 数学分支 | 调用/token | 有效 GFLOPs/token | 整体 kernel ms/token | 接口读+写 GB/token |
|---|---:|---:|---:|---:|
| Delta qkv/z | 72 | 3.019899 | 5.4016 | 3.0453 |
| MoE down W4 GMM | 48 | 1.572864 | 4.2658 | 3.6931 |
| HC down/up | 192 | 1.266156 | 3.0394 | 1.3028 |
| MoE gate/up W4 | 48 | 3.145728 | 2.0127 | 1.6085 |
| LM head W8 | 4 | 1.271398 | 1.7221 | 3.2280 |
| Delta output | 36 | 1.132462 | 1.7028 | 1.1418 |
| QSA q/k/v | 36 | 0.817889 | 1.4502 | 0.8285 |
| 共享专家 / router | 96 | 0.597934 | 1.3667 | 0.6051 |
| QSA output | 12 | 0.377487 | 0.5453 | 0.3817 |
| Delta beta/decay | 72 | 0.017695 | 0.4589 | 0.0192 |
| Indexer W8 | 12 | 0.039322 | 0.3381 | 0.1558 |
| Indexer FP32 score | 12 | 0.050332 | 0.1145 | 0.0284 |
| LM head mixer | 2 | 0.013107 | 0.0181 | 0.0138 |

| 额外主计算 | 有效 GFLOPs/token | kernel ms/token |
|---|---:|---:|
| 稀疏注意力 QK + AV | 0.604422144 | 0.4497 |
| FP32 DeltaNet 主递推 | 0.198180864 | 1.3306 |

**矩阵投影 13.271941120 GF + 索引打分 0.050331648 GF + 稀疏注意力 0.604422144 GF + DeltaNet 0.198180864 GF = 14.124875776 GF/token 的主要有效数学计算。**

具体算式：

- 普通投影：2×N×K；MoE 投影：2×10×N×K。全部投影形状、权重格式及字节下界见 `projection_flops_bytes.csv` / `inventory.json`。
- 索引打分：12×2×4×4096×128 = 50,331,648 FLOPs。
- 稀疏注意力：12×4×24×256×2049.5 = 604,422,144 FLOPs；2049.5 是本次 31 个周期的平均有效槽位数，固定物理宽度为 2051。按固定宽度计为 604,864,512 FLOPs。
- DeltaNet 主递推：36×5,505,024 = 198,180,864 FLOPs；原生代码计入显式加零等操作时为 198,623,232 FLOPs。

另有 RMSNorm、逐元素运算、卷积、激活和特殊函数；本报告未把 exp/rsqrt/sigmoid 等任意折算为“1 FLOP”。所以 14.125 GF 不是完整物理指令总量。Arithmetic 硬件计数器仅 stage0 成功导出，stage1 被外部模型切换打断，不能宣称已测得全模型的物理 FLOPs。

## 5. HBM：最低权重字节、接口 PMU、真实控制器流量

| 口径 | GB/token | 含义 |
|---|---:|---|
| 活跃权重及量化参数的逻辑字节下界 | 9.155820 | 假设每份有效权重/量化参数读取一次；不等于实际 HBM |
| AI Core/Vector main-memory 接口 PMU 读+写 | 20.957941 | 按 kernel 聚合，受接口层级、重复访问等影响 |
| **HBM 控制器真实读+写** | **14.608847** | 独立稳态窗口积分，两卡所有 HBM ID 求和 |
| └ HBM 读 | 11.035589 | 控制器计数 |
| └ HBM 写 | 3.573259 | 控制器计数 |

HBM 采样窗口选同一 HTTP 请求的 event8→event247，共 239 个 token 周期、10.395051 s，TPOT=43.493938 ms；两卡合计平均读写带宽 335.882 GB/s。每卡流量分别为 6.838595、7.770253 GB/token。

控制器采样粒度约 10 ms，无法把真实 HBM 字节精确分摊到微秒级 kernel。片上缓存、重复访问和计数范围会使接口 PMU 与 HBM 控制器数值不同；这三种口径不能相加或互换。HBM 系统级计数也包含同卡后台流量，不是逐进程独占计数；本轮未运行其他模型工作负载。

单位核验：本机 CANN 内部 `hbm.db` 的 bandwidth 是 MiB/s，由安装源码 `hbm_model.py` 的 `SCALE=32/2**20` 验证；乘 2²⁰ 后才是 B/s。按同一 HBM ID 的相邻采样区间积分，边界按覆盖比例裁切。CSV 的 Average 是各 HBM ID 的平均，不能当成整卡总带宽。

## 6. 固定开销和“纯计算 / 纯访存耗时”的边界

| 独立校准（每卡单进程） | stage0 对应卡 | stage1 对应卡 |
|---|---:|---:|
| 极小 AIV kernel，40 blocks，μs/次 | 0.279395 | 0.279797 |
| 256 MiB 双向大块 copy，读+写 GB/s | 1235.747 | 1239.769 |
| 4096³ BF16 GEMM，TFLOPS | 246.001 | 246.140 |

按两个阶段各自调用数乘极小 AIV kernel 时间，得到 **1.4380 ms/token 的调度底噪参考量**。它包含极小 kernel 本体与串行调度，不是每种 kernel 通用的固定成本，**已包含在实测时间中，不能另加到 43.3514 ms 上**。

按每卡实际 HBM 字节除以该卡大块 copy 校准带宽，再串行求和，得到 **11.8015 ms/token 的理想流式搬运参考量**。14.125 GF 除以约 246 TFLOPS 得到约 0.057 ms 的大 GEMM 等效参考；FP32、量化解包和 M=1 小矩阵不能沿用大 GEMM 吞吐，不能把这个值当成实际纯计算时间。

kernel 内计算和访存可以重叠，且还存在反量化、padding、低并行度、片上搬运和流水线等待。**仅凭当前计数器不能唯一确定一个“纯 FLOPs 时间 + 纯 HBM 时间 + 通用固定成本”的因果分解。** 可以严格闭合的是第 2 节时间区间账本；把所有差值塞入“固定开销”会产生错误结论。

## 7. 对下一步优化的直接含义

| 证据 | 下一步候选 | 可直接覆盖的现有耗时 |
|---|---|---:|
| M=1 投影多，普通 MatMul 是第一大类 | BF16 GEMV / 多投影融合，减少低利用率小矩阵和布局转换 | 14.097 ms（整类） |
| Gather 264 次、Transpose 96 次 | 专家布局常驻；W4 投影与专家选择、布局、归约协同设计 | Gather 3.707 + Transpose 2.059 ms |
| GMM down 与量化分支有大量接口写入 | 优化选中专家的 W4 GEMV，避免中间反量化/临时矩阵物化 | GMM 4.266 + WQBM 4.073 ms（两类） |
| Cast 858 次、Mul 615 次、Sigmoid 325 次 | 沿完整依赖链融合，减少调度与中间张量 | Cast 1.414 + Mul 2.277 + Sigmoid 0.577 ms |
| 所有非设备活动空隙合计仅约 2.118 ms | 宿侧优化作为辅助，主攻设备内核 | 即使全部消除也只减少约 4.89% TPOT |

这些类别彼此有交集，不能把各行候选收益相加。提高速度 20%（23.06→27.67 tokens/s）要求 TPOT 约 36.14 ms、减少约 7.22 ms；若要求 TPOT 本身降低 20%，则需约 34.68 ms、减少约 8.67 ms。都需要设备算子与数据布局的实质改动，单靠宿侧空隙不足以实现。

## 8. 证据、复现和服务状态

- 本地附件目录：`r136-decode-budget-evidence/`。`summary.json` 是机器可读结论；`time_budget.json`、`semantic_matrices.csv`、`kernel_types.csv`、`projection_flops_bytes.csv` 是核心表格。
- `runs.json` 保存完整周期时间戳；`traces/{timing,memory}_stage{0,1}/.../kernel_details.csv`、`task_time.csv` 保存原始两卡采样；`sys_hbm/.../hbm.db` 保存 HBM 计数。
- 本地可用 Python 执行 `analyze.py`、`semantic_ops.py`、`resources.py`、`integrate_hbm.py` 重新聚合。`calibrate_single.py` 是 NPU 侧可选复测脚本；`profile_model.py` / `profile_support.py` 依赖远程采样目录中的仪器化 fnx 副本，并非独立可运行的采集程序；不应覆盖生产服务。
- 首轮受模型争抢影响的 OOM 和双设备上下文错误校准均已剔除，未用于本报告。两张卡分别独立进程校准，无上下文错误；日志仅提示当前 CANN/HDK 需要 32 padding。
- 恢复核验：9002 `/health` 返回 healthy，revision=r135、ctx=262144、worker_alive=true；`model-switch.service` active、RefuseManualStart=no。未更改生产推理代码和部署参数。
- 远程原始证据保留在 `/models/fnx-diagnostics/r136-decode-budget-20261110/`。包含进程原始环境的私密文件未收入附件或 Git。

官方口径参考：[CANN AI Core 指标](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha001/maintenref/envvar/envref_07_0116.html)、[HBM 数据库说明](https://www.hiascend.com/document/detail/en/canncommercial/850/devaids/profiling/atlasprofiling_16_1144.html)。内部 hbm.db 单位以本机安装实现为准，不能直接套用合并数据库表的单位。
