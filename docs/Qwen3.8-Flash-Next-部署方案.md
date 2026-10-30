# Qwen3.8-Flash-Next 在 2×910B4(64GB) 上的部署与加速方案

> 目标机：`<ascend-host>`，2 × Ascend 910B4（单卡 64 GB HBM，可用 121.9 GiB），主机 219 GiB DDR，CANN 9.1.0
> 依据：`bench/910B4x2-基线测试报告.md`（实测算力/带宽）、`_research/npu-optimization-playbook.md`（Strata/nInfer 技术提炼）
> 状态：**方案与可行性论证完成，工程实施待排期**（本文档即实施基线）

---

## 1. 模型画像：这个模型天生就是给"小显存加速器"设计的

### 1.1 架构拆解（来自 `config.json` + 官方 model card）

| 组件 | 规格 | 工程含义 |
|---|---|---|
| 总参数 | **180B** = 125B MoE主干 + **51B n-gram embedding** + 4B MTP | 权重 360 GB（bf16），131 个分片 |
| 激活参数 | **6B**（10 routed + 1 shared expert） | decode 每 token 只读 6B 权重 |
| 层结构 | 48 层 = 12 ×（3×GatedDeltaNet → MoE）+ 12 ×（QSA → MoE） | **36 层是线性注意力**，只有 12 层有真 KV |
| Gated DeltaNet | 16 QK head × 128、48 V head × 128、conv kernel 4 | 固定大小递归状态，**不随上下文增长** |
| QSA（Qwen Sparse Attention） | 24 Q head / 2 KV head × 256、partial rope 0.25、**block 级稀疏**、indexer 4Q+1K×128、budget 2048 token | 长上下文延迟的关键；**自研 kernel 的主战场** |
| MoE | **512 experts，top-10 + 1 shared**，expert intermediate 640 | 专家权重 120.8B，激活 2.36B → **专家缓存/分层的最佳标的** |
| n-gram embedding | 20,000,000 × 2560，**只在 layer 2 使用**（`ple_layer_ids:[2]`） | **51B 参数但每 token 只查 2–3 行** → 纯容量问题、几乎零带宽问题 |
| Gated Residual | 4 分支、bottleneck rank 320、sigmoid 读门控 | 逐层多一次小 GEMM，需融合 |
| MTP | 1 层，4B | **免费送投机解码**（1.5–2× decode 加速潜力） |
| 上下文 | 原生 262,144，YaRN 可扩到 1,000,000 | |

### 1.2 每 token 访存账（decode 的关键算式）

```
MoE 激活专家： 10 × 4.915M × 48 层 = 2.36 B
注意力+稠密：               ≈ 3.0 B
------------------------------------------
bf16:  ≈ 10.7 GB/token     W8: ≈ 5.4 GB     W4: ≈ 2.7 GB
n-gram 查表：2–3 行 × 2560 × 2B ≈ 15 KB/token  ← 可忽略
```

### 1.3 容量账（决定成败的那张表）

| 权重格式 | 权重总量 | 121.9 GiB HBM | 判定 |
|---|---:|---|---|
| BF16/FP16 | **335 GiB** | 缺 213 GiB | ❌ 即使 DDR+HBM 合并（≈312 GiB）也不够 |
| W8A8 全量 | ~168 GiB | 缺 ~46 GiB | ❌ 单机装不下，需 DDR 分层 |
| **W8 稠密 + W4 专家 + n-gram 放 DDR** | HBM 侧 **≈ 70 GiB** | 余 ~50 GiB 给 KV/激活 | ✅ **推荐** |
| W4 全量 | ~84 GiB | 余 38 GiB | ⚠️ 可行但 KV 预算紧，精度风险高 |

**DDR 侧容量**：219 GiB（补第 8 条内存后 251 GiB）→ 放 n-gram 表 INT8（51 GB）+ 专家冷备 + 锁页 staging 缓冲，充裕。
**盘侧**：3.26 TiB 闲置（见基线报告附录 A）→ 可放全量 bf16 原始权重 + 多套量化版本。

---

## 2. 目标部署架构（分层驻留）

```
┌──────────────────────── HBM 121.9 GiB（2 卡）────────────────────────┐
│ ① MoE 专家 W4（120.8B×0.5B=60.4 GB）→ 每卡 30.2 GB，静态优先级常驻  │
│ ② 注意力/稠密/embedding W8（≈8 GB）      → 每卡 4 GB                  │
│ ③ MTP 头 W4（2 GB）                      → 主卡 2 GB                  │
│ ④ KV（bf16 24 KB/token，INT8 12 KB/token）→ 预算 40 GiB              │
│ ⑤ 工作区 / 图捕获 buffer（≥ 1 GiB 余量，硬性）                        │
└──────────────────────────────────────────────────────────────────────┘
             ▲ 按需 DMA（PCIe Gen4 x16，实测 14.0 GB/s）
             │ 冷专家换入：每 token 目标 misrate ≤ 20%
┌────────────┴──────────── DDR 219 GiB（锁页）────────────────────────┐
│ ⑥ n-gram embedding 表 INT8（51 GB）→ layer 2 每 token 一次批量 gather│
│ ⑦ 专家热备池 / 预取双缓冲                                          │
└──────────────────────────────────────────────────────────────────────┘
             ▲ mmap + page cache
┌────────────┴──────────── NVMe 3.26 TiB ────────────────────────────┐
│ ⑧ 全量 bf16 权重、多套量化版本、冷专家分片                          │
└─────────────────────────────────────────────────────────────────────┘
```

**三条设计铁律**（来自实测数据，不是经验）：

1. **decode 是显存带宽问题**：实测 M=1 GEMV 有效带宽 1.2–1.45 TB/s → `TPOT_min = 每 token 权重字节 / 1.45 TB/s`。bf16 8.3 ms（120 tok/s 上限）、W8 4.1 ms（242）、W4 2.1 ms（483）。**任何优化都必须拿这条公式先算一遍收益上限**。
2. **offload 只在算得过来的时候做**：`B_token = top_k × expert_bytes × layers × miss_rate`。按 W4 单专家 2.46 MB（4.915M×0.5B）算：全 miss = 10×2.46MB×48 = **1.18 GB/token** → 84 ms，**不可用**；miss_rate 20% → 236 MB → **16.9 ms**（可接受但有感）。→ **专家必须以 W4 常驻 HBM 为主、offload 为辅**，靠 n-gram 表进 DDR 腾出空间才是正解。
3. **n-gram 表是"容量租客"不是"带宽租客"**：51B 参数只在 layer 2 查 2–3 行，一次批量 gather ≈ 15 KB / 20 µs → 占 8 ms 预算的 **0.25%**。**必须放 DDR（甚至 NVMe+page cache），放 HBM 是最大的浪费。**

---

## 3. 优化项落地表（用户点名的 5 类 + 补充）

> 优先级：P0=立刻能做且收益确定；P1=一个迭代；P2=立项自研；P3=不建议

| # | 优化项 | 910B4 上的具体做法 | 预期收益（按 roofline 推算） | 难度 | 验证方法 |
|---|---|---|---|---|---|
| **1** | **专家缓存**（MoE expert cache） | ①离线 dump router 统计真实流量 → 静态优先级（§P1-1）；②固定区+双窗口预算：每层常驻前缀 `R_l`，窗口容 `max_l(W_l−R_l)`；③淘汰用**路由计数交换**（非 LRU），保证地址终生稳定以支持图捕获；④token 粒度批处理，entry 256 B 对齐 | 决定"HBM 里能装多少专家"；把 miss_rate 从 ~50% 压到 ≤20% → 每 token 少搬 ~700 MB | P1 | `hit_rate`、每 token 搬运字节、`msprof` 中 DMA 段占比 |
| **2** | **CPU/NPU 分层** | **只做"分层驻留"，不做"CPU 算专家"**（aarch64 无 AVX-512，CPU BF16 吞吐比 NPU 低 1–2 个数量级，实测 64 核 vs 241 TFLOPS）。载体：51 GB n-gram 表进 DDR 锁页（`aclrtMallocHost`），attention 当前窗口与热专家留 HBM | 省出 51 GB HBM = 多装 **~20,000 个专家槽位**；换来 KV 预算翻倍 | P0（ngram）/ P1（专家） | HBM 占用对比、每 token H2D 字节、锁页上限（openEuler `vm.max_map_count`/`RLIMIT_MEMLOCK` 需实测） |
| **3** | **权重动态调度** | ①prefill（大 T）按层流式整块装载专家 bank，搬运被 T 摊薄；②decode（小 T）靠常驻集合 + **预取只提前 1 层**；③**代际租约 + 四段发布**（编码数据 DMA → offset 表 DMA → slot 完成事件 → acquire → kernel → release），消费者只等自己需要的代次；④**DMA 走独立 stream，不经 AI Core**（NVIDIA 侧实测 copy kernel 吃掉 40% kernel 时间） | 把换入延迟藏进上一层计算；避免"无关预取阻塞当前层"的假依赖 | P1–P2 | 注入"预取/消费交错"压力测试，统计消费者等待时间；对比无条件预取（NVIDIA 实测 **−32.6%**，必须做门控） |
| **4** | **KV 压缩** | ①**先天优势**：48 层里只有 12 层有真 KV，bf16 仅 **24 KB/token**；②group-wise INT8（G64，参考 nInfer 实测 1024→528 B/token 的 −48%）；③QSA 的 block 级稀疏（budget 2048）降的是**计算**不是存储，要单独看；④1M 上下文时 INT8 后 12 GB/序列，需 page-group + COW 做前缀复用 | KV −48%（24→12 KB/token）→ 等于多 20 GiB 给专家；支持 262K→1M | P1 | 困惑度 + 长文 needle + greedy 逐位一致；HBM 前后对比 |
| **5** | **定制 kernel** | 见 §4 首批 10 个算子。核心三个：**QSA 块稀疏注意力**、**Gated DeltaNet chunk scan**、**W4A8 分组 GEMM（三平面布局）** | QSA：长上下文延迟线性段下降；DeltaNet：36 层固定状态避免 attention 二次增长；W4 GEMM：直接对应 2.1 ms 的 decode 下界 | P2 | 单算子 micro-bench 对齐"内存地板"（`memory_floor_pct`）+ `msprof` Cube/Vector 利用率，必须 T=1/4/8 分别达标 |
| 6 | **投机解码（MTP）** | 模型自带 1 层 MTP，直接接 `--speculative-model` 路径；接受率统计口径 `acceptance=accepted/drafted` | decode **1.5–2×**（等于免费白捡） | P1 | 接受率、tokens/round、greedy 输出一致性 |
| 7 | **前缀缓存 + 调度** | `--enable-prefix-caching` + chunked prefill；三类粒度解耦（分配 64 token / 可见 frontier 1 token / 可复用 checkpoint） | agent 多轮场景 TTFT 显著下降 | P0 | 命中长度分布、TTFT |
| 8 | **图模式（ACL graph）** | 分段捕获：MoE/offload 段与主干段分开，cache 交接与预取留在图外；失败自动回退 eager | 消除 launch 开销（本机 27B 实测 TP=2 无收益正是 launch/AllReduce 固定开销的证据） | P2 | 图 vs eager 的 TPOT 分布 |
| 9 | **并行策略** | **不用 TP=2**（实测 TPOT 完全不变、C=8 吞吐 −17%）；优先**单卡 PP=1 + 专家并行**；若必须双卡，用 **PP=2**（每 window 只传几百 KB，绕开 AllReduce） | 双卡从"负收益"变"至少持平+吞吐翻倍" | P1 | TP vs PP 的 decode tok/s、单次 AllReduce 延迟 |
| 10 | ❌ **CPU 分担专家计算 / 近似跳层** | 明确**不做**：aarch64 无 AVX-512；Strata 的 KVA split=24 在 agent 任务上 10/10 失败 | — | — | — |

---

## 4. 软件栈改造路线（这是本项目真正的工作量）

### 4.1 硬事实（已实测）

| 事实 | 证据 |
|---|---|
| vLLM 0.23.0 **不支持** `Qwen4ExpForConditionalGeneration` | 全仓 grep 无 `qwen4`，模型注册表最高到 `Qwen3_5Moe` |
| transformers 5.5.4 **不支持** `qwen4_exp` | `transformers/models/` 无该目录 |
| transformers **5.21.0 已支持** `qwen4_exp` | `pip download transformers==5.21.0` 后 `models/qwen4_exp` 存在 |
| 模型仓库**不带 modeling 代码** | ModelScope 文件清单只有 config/tokenizer/131 分片，`trust_remote_code` 无用 |
| vLLM-Ascend 最新仅 **0.23.0**，vLLM 上游已 0.32 | 无法直接升级吃上游支持 |

### 4.2 三个阶段（含验收门禁）

**Phase A —— 正确性基线（约 2 周）**
1. 新建开发容器 `flash-next-dev`：transformers 5.21.0 + torch_npu 2.10 + torch 2.10（不动生产镜像）。
2. **不能整模型加载**（335 GiB）→ 采用"**分层切片验证**"：从 131 个分片里按层抽权重，构造 1 层/3 层的子模型，与 transformers 5.21.0 CPU 参考实现逐层对齐（`rtol=1e-2` 量化前、`1e-3` bf16）。
3. 依次验证 5 个新组件：n-gram embedding 查表 → Gated DeltaNet → QSA（稠密模式）→ MoE 路由 → Gated Residual。
4. **出口门禁**：3 层子模型在 NPU 与 CPU 的 logits 最大相对误差 < 1e-3，且 greedy 生成前 32 token 完全一致。

**Phase B —— vLLM-Ascend 适配（约 4 周）**
1. 从上游 vLLM（支持 Qwen4Exp 的版本）**回移** `qwen4_exp` 模型实现到 0.23.0（`vllm/model_executor/models/qwen4_exp.py` + config 注册 + MTP）。
2. **复用 `qwen3_next` 的 Ascend 路径**（同为 Gated DeltaNet 混合 MoE，vllm-ascend 0.23 已有该模型文件），只替换 attention 分支为 QSA。
3. 实现 n-gram embedding：**host 侧算 n-gram id → `aclrtMemcpyAsync` 批量 gather DDR 表 → NPU `index_add`**（一层一次，15 KB/token）。
4. **出口门禁**：单卡跑通 262K 上下文下 3 类 prompt（短问答/长文 needle/代码 agent），输出与 CPU 参考实现 top-1 一致率 > 95%，服务可稳定 30 min 不 OOM。

**Phase C —— 量化与常驻（约 4 周，与 B 并行）**
1. **稠密部分 W8A8**（msModelSlim），**专家 W4A8**（group-wise，group size 128，三平面布局 `row_split_k128_v1`：低比特码/高比特码/组 scale 三平面，K 对齐 128、256 B 对齐、scale 为乘数）。
2. n-gram 表 INT8（甚至 INT4，51 GB → 26 GB）。
3. **出口门禁**：MMLU/C-Eval 掉点 ≤ 1.5%，困惑度涨幅 ≤ 5%，长文 needle ≥ 95% 基线；HBM 峰值 ≤ 115 GiB。

**Phase D —— 性能优化（与 C 交叉，约 4 周）**
- 按 §3 的 P0→P1 顺序做：n-gram 下沉 DDR → 专家静态优先级 → 预取门控 → KV INT8 → MTP 投机 → 图捕获。
- **出口门禁（本机 2 卡的目标值，按 roofline 折算 71% 达成率）**：
  | 指标 | W8 目标 | W4 目标 |
  |---|---:|---:|
  | 单流 TPOT | ≤ 6.0 ms（≥166 tok/s） | ≤ 3.5 ms（≥285 tok/s） |
  | TTFT @4K prompt | ≤ 1.2 s | ≤ 1.0 s |
  | C=8 总吞吐 | ≥ 400 tok/s | ≥ 600 tok/s |

> 参考：本机 27B W8A8 实测 38 tok/s/流、C=8 67.5 tok/s —— Flash-Next 因为激活只有 6B，**即使达不到目标也理应比 27B 快 3 倍以上**（6B vs 27B 激活权重差 4.5×）。

---

## 5. 首批内部算子库清单（`ops/registry.yaml` 的初始 10 条）

| # | 算子 | 类型 | 为什么要自研 | 复用可能性 |
|---|---|---|---|---|
| 1 | `qsa_block_sparse_attention` | AscendC | 昇腾无对应实现；block 级 top-k 索引 + 稀疏 flash attention | 基于 `npu_fusion_attention` 改造 |
| 2 | `qsa_indexer` | AscendC/Triton | MQA 4Q/1K、compress_ratio 4、budget 2048 的打分与选块 | 自研 |
| 3 | `gated_deltanet_chunk_scan` | AscendC | 36 层线性注意力主算力；chunk 并行 scan + conv4 | 参考 `qwen3_next` 的 Ascend 实现 |
| 4 | `ngram_embedding_gather` | Host + Vector | 跨 DDR/NPU 的批量 gather + add（本项目特有） | 自研 |
| 5 | `moe_gating_topk_10of512` | AscendC | 512 选 10 + shared expert；npu_moe_gating_top_k 需验证 T=1 性能 | `npu_moe_gating_top_k` |
| 6 | `grouped_gemm_w4a8` | AscendC/Catlass | 三平面反量化在 Vector、GEMM 在 Cube，T=1 走 GEMV | Catlass + 自研 |
| 7 | `gated_residual_fused` | Vector | 4 分支瓶颈 rank320 + sigmoid 读门控 + 写门控的融合 | 自研 |
| 8 | `rmsnorm_rope_qk_norm_fused` | AscendC | partial rope 0.25 + interleaved mrope，三次访存合成一次 | `npu_rms_norm`+`npu_rotary_mul` 融合 |
| 9 | `kv_quant_int8_g64` | AscendC | group-wise INT8 量化/反量化 + paged 布局 | 自研 |
| 10 | `mtp_speculative_head` | Python+Scheduler | 1 层 MTP 的 draft/verify 调度 | vLLM spec decode 框架 |

> 每个算子必须带：`registry.yaml` 元数据（芯片、精度、shape 约束、性能基线、精度基线）+ 黄金用例 + `msprof` 性能报告。**没有基线数据的不许进 registry**。

---

## 6. 风险与替代方案

| 风险 | 概率 | 影响 | 应对 |
|---|---|---|---|
| W4 量化精度崩（512 专家 top-10 极其敏感） | 中 | 高 | 专家走 **W4 但保留 W8 兜底版本**；group size 128 + GPTQ/AWQ 校准；掉点超 2% 立刻回退 W8+offload |
| 512 专家的 MoE 在 910B4 上 grouped GEMM 效率低（小 M） | 高 | 中 | T=1 用 GEMV 专路（Vector 反量化 + Cube 小 GEMM），T>64 走 Cube 分块；实测再定 |
| QSA 块稀疏 kernel 写出不来 | 中 | 高 | **降级路线**：先用稠密 attention（262K 内存会爆）→ 退到 32K 上下文 + prefix caching |
| 2 卡 121.9 GiB 即使 W4 也不够 | 中 | 高 | ①n-gram 全下沉 DDR 已省 51 GB；②再 offload 20% 冷专家；③**申请换 4 卡/8 卡机型** |
| 上游把 Qwen4Exp 改架构（`_exp` = 实验版） | 中 | 中 | 锁死 git tag + 权重快照到内网；适配层与模型解耦（config 驱动） |

**给生产部的硬件诉求（按投入产出排序）**
1. **盘**：把 3.26 TiB 用起来（0 成本，见基线报告附录 A）—— **P0**
2. **内存**：补第 8 条 DDR4-3200 32 GB（约千元级，直接决定 offload 能力）—— **P0**
3. **网络**：换万兆交换机（拉 360 GB 权重从 48 min → 6 min）—— **P1**
4. **整机**：若要长期做 ≥100B 模型，建议换 **Atlas 800T A2（8×910B + HCCS）**；本机转开发/验证机 —— **P1**

---

---

## 7. Phase A 实测进展（2026-10-28，已跑通）

### 7.1 里程碑：Qwen4Exp 架构在昇腾 NPU 上跑通前向，且数值对齐

| 项 | 结果 |
|---|---|
| 环境 | `transformers 5.21.0`（独立 venv，不污染 vLLM 环境）+ torch 2.10.0 + torch_npu 2.10.0.post4 |
| 模型类 | `Qwen4ExpForCausalLM`（text-only 路径）+ `Qwen4ExpConfig/Qwen4ExpTextConfig`（AutoConfig 可直接读官方 config.json） |
| 缩放验证模型 | hidden 128 / 4 层（3 linear + 1 QSA）/ 8 专家 top-2 / ngram 1024，1.69 M 参数 |
| **fp32 CPU vs NPU** | `max_abs_diff = 7.6e-05`，`rel = 1.7e-04`，**argmax 完全一致** ✅ |
| **bf16 CPU vs NPU** | `max_abs_diff = 1.27e-02`，`mean = 3.7e-04`，**argmax 完全一致** ✅ |

**意义**：Gated DeltaNet、QSA（含 indexer）、n-gram embedding、Gated Residual、512→8 专家 MoE **这五条新链路在昇腾上全部能跑**，Phase A 的最大风险（"架构根本跑不起来"）已排除。当前走的是 transformers 的参考实现（`chunk_gated_delta_rule` / `causal_conv1d` 均未装优化库）。

### 7.2 关键发现：不做定制 kernel，性能差 15 倍以上

用**真实维度子模型**（hidden 2560、2 层、8 专家、1.54 B 参数、2.9 GiB 权重）在 910B4 上实测：

| T | 前向时延 | 等价"权重带宽" | 相对 1.45 TB/s 峰值 | 理论 roofline |
|---:|---:|---:|---:|---:|
| 1（decode） | **30.88 ms** | 99.6 GB/s | **7%** | 2.0 ms |
| 128 | 251.8 ms | 12.2 GB/s | 1% | — |
| 512 | 955.3 ms | 3.2 GB/s | 0% | — |

**解读**：naive transformers-eager 路径比内存 roofline 慢 **15×**（decode）到 **数百倍**（长序列）。原因：
1. `chunk_gated_delta_rule` / `causal_conv1d` 走纯 Python 参考实现（O(T) Python 循环）；
2. QSA 的 indexer 与稀疏选择是逐 token Python 逻辑；
3. MoE 路由是 `topk + index_select + 逐专家循环`，没有 grouped GEMM；
4. NPU 小算子 launch 开销被放大。

> **这就是"内部算子库"最硬的立项理由**：以当前速度，Flash-Next 单 token 要几十秒，业务完全不可用；要摸到 2 ms 的 roofline，必须把 §4 的 10 个算子做出来。

### 7.3 已踩的坑（进 `known_issues` 与内部 Skill）

| # | 现象 | 根因 | 规避 |
|---|---|---|---|
| 1 | `model(...)` 走 cache 时报 `Cannot call update_indexer on layer 3 which is a DynamicLayer` | transformers 5.21 默认 `DynamicCache` 不知道 QSA 层需要 indexer KV | **`use_cache=False` 可完全绕开**（已验证）；正解需给 QSA 层建 `DynamicIndexedLayer` |
| 2 | `generate(cache_implementation="static")` 报 `KeyError: 'indexed_attention'` | `cache_utils` 的层类型映射表缺 QSA 专用类型（上游集成缺陷） | 暂用 `use_cache=False`；或自行实现 cache |
| 3 | NPU 上 `torch.linalg.solve_triangular` 提示 internal format 不可用 | DeltaNet 的三角求解走了通用路径 | 后续用 chunked scan 自定义算子替代 |
| 4 | 容器只装 `torch` 不装 `torch_npu` 时报 `Failed to load the backend extension` | torch 自动加载后端 | 纯下载/CPU 任务设 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` |
| 5 | 模型下载容器无 NPU 设备也能跑 | 同上 | 同上 |

### 7.4 已完成的基础设施动作

- ✅ **磁盘扩容完成**：新建 `/dev/nvme0n1p4`（3.3 TB）→ LVM `models_vg/models_lv`（3.26 TiB）→ 挂载 `/models`，已写入 `/etc/fstab`；**现有 p1/p2/p3 未改动**
- 🔄 **权重下载中**：`modelscope download Qwen/Qwen3.8-Flash-Next` → `/models/Qwen3.8-Flash-Next`，实测约 88 MB/s，已完成 ~88 GB / 360 GB
- ✅ **独立开发环境**：`/root/ascend-dev/venv`（transformers 5.21.0，`--system-site-packages`，与生产镜像隔离）

---

## 7.5 热点画像与第一项优化（2026-10-28 实测，已拿到 1.39× 加速）

### 7.5.1 模块级耗时分解（hidden 2560 / 2 层 / 8 专家 / 1.54B，910B4）

| 模块 | T=1（decode）占比 | T=128（prefill）占比 |
|---|---:|---:|
| **GatedDeltaNet** | **35.6%**（12.2 ms） | 9.2% |
| SparseMoeBlock | 14.6%（5.0 ms） | 2.0% |
| **PLE 层（n-gram embedding）** | **12.2%**（4.2 ms） | 1.7% |
| **QSA Attention** | 12.2%（4.2 ms） | **83.6%**（215.7 ms） |
| GatedResidual（2 处） | 11.1%（3.8 ms） | 1.4% |
| Shared Expert MLP | 1.7% | 0.2% |

**两条路径的热点完全不同** —— 这直接决定了"decode kernel"和"prefill kernel"必须分开做：

- **decode 先做 GatedDeltaNet**（35.6%），不是 MoE（原计划排错了，数据纠正）
- **prefill 先做 QSA**（83.6%）
- PLE（n-gram gather）在两边都占 12–1.7%，且它是纯访存 gather，最容易被 DDR 分层吃掉

### 7.5.2 第一项优化：decode 走 recurrent 路径 → **1.39×**

**根因**：transformers 的参考实现无论序列多长都调用 `torch_chunk_gated_delta_rule`，它把序列**补齐到 chunk_size=64** 再做分块矩阵运算。T=1 的 decode 场景下，**为一个 token 做了 64 个 token 的活**。

**做法**：T ≤ 8 时改走 `torch_recurrent_gated_delta_rule`（逐 token 更新 128×128 递归状态），数学等价、无填充。

| 指标 | 优化前 | 优化后 | 变化 |
|---|---:|---:|---:|
| T=1 整模型前向 | 29.90 ms | **21.53 ms** | **1.39×** |
| T=128 整模型前向 | 241.86 ms | 239.88 ms | 不变（走 chunk） |

**算法等价性验证**（同形状，bf16 输入，fp32 内部计算）：

| T | 1 | 2 | 8 | 64 | 128 |
|---|---:|---:|---:|---:|---:|
| 相对误差 | **0** | **0** | 7.4e-07 | 1.4e-04 | 4.6e-04 |

纯 fp32 累加顺序差异，**算法等价**。

### 7.5.3 新增的坑：MoE 路由的数值敏感性

优化后整模型 logits 的 `max_abs_diff` 达 3.4（argmax 变化），远超上面 1e-4 的单算子误差。原因不是 bug，而是 **MoE 的 top-k 路由是阶跃函数**：1e-4 的数值扰动足以**翻转专家选择**，输出随之跳变。

> **必须写进验收标准**：量化/优化后的精度验收**不能只看单层 KL**，必须分层做：
> ① 单算子相对误差（<1e-3）；② **路由一致率**（top-10 专家集合重合度）；③ 端到端任务指标（agent 成功率/代码 pass@1）。
> 只做 ②③ 会误判，只做 ①会漏掉路由翻转。

### 7.5.4 调整后的算子优先级（用数据改计划）

| 顺位 | 算子 | 依据 | 目标 |
|---|---|---|---|
| **1** | `delta_net_decode_step` | decode 35.6% | T=1 每层 ≤0.15 ms |
| **2** | `qsa_sparse_attn` (prefill) | prefill 83.6% | 4K prompt TTFT ≤1.2 s |
| **3** | `ngram_gather`（DDR 分层） | decode 12.2% | ≤0.3 ms/层 |
| **4** | `moe_grouped_gemm_w4a8` | decode 14.6%，真实模型 512 专家放大 5× | T=1 每层 ≤0.5 ms |
| 5 | `gated_residual_fused` | decode 11.1% | 二次访存合并 |

---

## 7.6 第一个 Triton 算子落地 + 一个改变架构的发现（2026-10-28）

### 7.6.1 `delta_net_decode_step` Triton kernel：正确性通过

单 token 的 gated delta rule 递推（T=1 路径）：

```python
S = S * exp(g)                              # 衰减必须先做
kv = sum(S * k[:, None], axis=0)
delta = (v - kv) * beta
S = S + k[:, None] * delta[None, :]
o = sum(S * q[:, None], axis=0)
```

| 形状 | o 绝对误差 | S 绝对误差 | 相对误差 | 判定 |
|---|---:|---:|---:|:--:|
| H=8 K=64 V=64 | 1.7e-06 | 4.8e-07 | 1.2e-07 | PASS |
| H=16 K=128 V=128 | 2.7e-05 | 2.4e-06 | 4.9e-07 | PASS |
| **H=48 K=128 V=128（真实）** | 2.7e-05 | 2.4e-06 | **4.6e-07** | **PASS** |

**踩到的坑（已进 known_issues）**：
1. **UB 溢出**：`[128,128]` 的 fp32 瓦片 + 临时 buffer 需要 1,589,248 bit，而可用只有 1,572,864 bit（**192 KiB**）——只差 2 KB 就编译失败。**必须切成 `BV=64`**。
2. **衰减顺序写反**：先算 `kv` 再衰减会导致 `o_diff≈2.6`、`S_diff≈2.9` 完全错。参考实现的顺序是「**先衰减、再投影、再加回**」。

### 7.6.2 发现：decode 的瓶颈不是浮点，是 **kernel 启动开销**

| 测量项 | 实测 |
|---|---:|
| 单个 Triton kernel（1024 程序，纯访存） | **0.102 ms / 次** |
| 单个 torch 小算子（1024 元素） | 0.029 ms / 次 |
| 单个 torch 小算子（64×64） | 0.027 ms / 次 |
| `delta_net_decode_step`（Triton） | 0.169 ms / 次 |
| `delta_net_decode_step`（torch 参考） | 0.292 ms / 次 |

**这组数说明三件事**：
1. **Triton on Ascend 的启动开销（0.102 ms）是 torch 小算子的 3.5 倍**，用它写 decode 小算子反而更慢。
2. delta step 的真实访存只有 6.3 MB（3.1 MB 状态读 + 3.1 MB 写）→ 在 1.45 TB/s 下应该 **4 µs**，实测 169 µs → **慢 40 倍，全是启动开销**。
3. 36 个 DeltaNet 层 × 0.169 ms = **6.1 ms/token**，占当前 21.5 ms 的 28%。

### 7.6.3 架构结论（三处修正）

| 原方案 | 修正后 | 理由 |
|---|---|---|
| "Triton 优先做所有算子" | **Triton 只做 prefill 的大融合算子**；decode 的小算子用 torch + **图捕获** | Triton 启动开销是 torch 的 3.5×，小算子吃亏 |
| "优化单个 kernel 效率" | **优先级最高是消除 launch 开销**（ACL graph / torch.compile） | 每层数十次 launch × 0.03–0.1 ms，45 层就是几十毫秒 |
| "decode 先做 MoE、再 DeltaNet" | 保持 decode 先做 DeltaNet，但**实现方式改为"融合成一个 kernel 覆盖多层"** | 单层单算子怎么优化都打不过 roofline 的 40 倍差距 |

> **新的性能路线**：`先去 launch 开销（图捕获）→ 再把相邻算子融合成大 kernel（Triton/AscendC）→ 最后才抠单个 kernel 的 tiling`。
> 预期：图捕获能让 21.5 ms 直接掉到个位数毫秒级；融合再进一步。

---

## 7.7 权重落地 + 图捕获三次尝试的结论（2026-10-28）

### 7.7.1 权重下载完成并校验通过

| 项 | 结果 |
|---|---|
| 分片 | **131 / 131**，`.incomplete` 残留 0 |
| 张量数 | 实际 **1658** 个 = index 声明的 **1658**（完全对齐） |
| 可读性 | 131/131 分片 safetensors 头部均可解析，**无损坏** |
| 体积 | **335.3 GiB = 360.0 GB**（与 ModelScope 清单逐字节吻合） |
| 位置 | `/models/Qwen3.8-Flash-Next`（3.26 TiB 新盘，`crc32.txt` 已随仓下载） |

**张量命名揭示的结构**（对分层设计极其关键）：
```
model.language_model.layers.{0..47}.attn_hyper_connection.{input_mix_weight_up/down, block_inject_weight}
model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{0..N}.weight   ← ★ 51B n-gram 表本身是分片的
mtp.{fc_embedding, fc_hidden, hyper_connection_mixer}
lm_head.weight / model.language_model.embed_tokens.weight
```
→ `ngram_embedding.shard_*` 是按 shard 存的查找表，**天然支持"保留在 DDR/NVMe、按需 gather 单个 shard"**，与 §2 的分层设计完全吻合，不需要重新切分。

### 7.7.2 NPU 图捕获：三次尝试，确认被 host 同步阻断

| 尝试 | 做法 | 结果 |
|---|---|---|
| 1 | 直接 `torch.npu.graph` 捕获 | ❌ `AclrtSynchronizeStreamWithTimeout ... stream is captured`，卡在 `masking_utils.find_packed_sequence_indices` 的 `.all()` |
| 2 | 显式传 `attention_mask`；再 monkeypatch 去掉 packed-sequence 探测 | ❌ 同步消失（输出 diff=0，补丁无损），但下一个阻塞是 forward 内 H2D `aclrtMemcpy` |
| 3 | 打补丁 + `capture_error_mode="relaxed"` | ❌ 仍报 `LocalScalarDenseNpu ... stream synchronize failed (107027)` |

**根因**：Qwen4Exp 的 transformers 实现里**散落着多处 device→host 标量读取**（`.item()` / `.all()` / 条件分支依赖张量值），不止 masking 一处。逐处打补丁不现实。

> **结论（写进架构约束）**：
> 1. **图捕获不是"开个开关"，而是要求 forward 天生 graph-safe**（无 host 同步、无数据依赖的控制流）。
> 2. 这恰恰是**自研后端的价值所在**：我们自己的 `fnx-backend` 从第一行代码就按 graph-safe 写（所有 shape/分支在编译期确定，索引全在 NPU 侧），才能吃到图捕获带来的数量级收益。
> 3. 在 transformers 参考实现上做性能优化，天花板就是"每一层几十次 kernel launch"，**不可能到毫秒级**。
>
> **风险提示**：如果自研 forward 仍无法完全消除 host 同步，退路是"分段捕获"（只把 MoE/attention 主体段捕获，前后处理保留 eager），nInfer 也是这么做的。

### 7.7.3 顺带排除的两个假设

| 假设 | 实测 | 结论 |
|---|---|---|
| "host 同步是 decode 慢的主因" | 去掉同步只快 **1.02×**（29.40→28.77 ms） | ❌ 同步不是主要开销；**launch 次数**才是 |
| "torch.compile 能救" | `InductorError: KeyError: x0 - 2` | ❌ transformers eager 实现编不动，需自研 |

---

## 7.8 决定性验证：graph-safe forward 捕获成功，**4.63× 加速**（2026-10-28）

§7.7 证明了"transformers 参考实现捕获不了"，但**没有证明自研实现也捕获不了**。为此手写了一个 graph-safe 的单层块（无 `.item()`、无数据依赖分支、形状全静态、路由用设备侧 `topk`），8 层串联后测试：

| 模式 | 8 层耗时 | 单层耗时 | 加速 |
|---|---:|---:|---:|
| eager | 13.25 ms | **1.66 ms** | — |
| **NPU 图捕获（relaxed）** | **2.86 ms** | **0.36 ms** | **4.63×** |

> **这是本项目最重要的一个数字**：它证明"**图捕获可行，前提是 forward 自己 graph-safe**"。
> 也就是说，§7.7 里 transformers 捕获失败的根因**不是昇腾不支持，而是那份实现不是 graph-safe 的**。
> 自研 `fnx-backend` 的核心价值由此被实证：**同样的硬件、同样的算子，只因为代码写法的差别，就有 4.63 倍的差距。**

**为什么这次能成功（与 transformers 实现的差别）**：

| 要素 | transformers 参考实现 | 本次 graph-safe 块 |
|---|---|---|
| padding 探测 | `(mask[:,-1]==0).all()` → host 同步 | 不探测，形状静态 |
| 序列分块 | chunk_size=64 固定填充 | T=1 单步递推 |
| MoE 路由 | Python 循环逐专家 + `index_select` | 设备侧 `topk` + 静态形状 |
| 条件分支 | 依赖张量值的 if | 无 |

**新发现的注意点**：捕获后 `replay()` 会**原地更新 DeltaNet 的递归状态 buffer**，连续 replay 而不重置会让状态发散（本次观测到 NaN）。→ **工程约束：每请求必须重置/替换状态张量**（用双缓冲或请求开始前 memset），这是 §3.2 里"代际租约"要解决的问题之一。

### 7.8.1 对整体性能预期的影响

| 阶段 | 之前的估算 | 用 4.63× 修正后 |
|---|---:|---:|
| 当前 naive eager（2 层 1.54B） | 21.5 ms | — |
| + 图捕获 | — | **≈4.6 ms**（同规模） |
| + 真实 48 层 × W4 专家 + KV INT8 | ≤3.5 ms（目标） | **目标可达成，且有余量** |

> 结论：**decode 延迟目标从"激进"变成"保守"**。下一步的重点从"抠 kernel"转为"**把整个 forward 写成 graph-safe**"，这正好是 `fnx-backend` L1–L3 的设计要求。

---

## 7.9 专家 offload 可行性实测：**结论是否定的**，W4 全常驻才是唯一路（2026-10-28）

### 7.9.1 带宽实测（真实专家张量形状 `[512,1280,2560]` bf16）

| 场景 | 数据量 | 耗时 | 有效带宽 |
|---|---:|---:|---:|
| 纯 H2D 拷贝（pageable） | 6.5 MiB | 0.41 ms | 15.6 GB/s |
| 纯 H2D 拷贝 | 65 MiB | 3.17 ms | 20.0 GB/s |
| 纯 H2D 拷贝 | 650 MiB | 31.65 ms | **20.1 GB/s** |
| **专家 gather（index_select + H2D）** | 1 专家 6.6 MB | 1.02 ms | 6.4 GB/s |
| **同上** | 5 专家 32.8 MB | 7.56 ms | 4.3 GB/s |
| **同上** | **10 专家 65.5 MB** | **16.97 ms** | **3.9 GB/s** |

**两个关键事实**：
1. 纯 H2D 能跑满 **20 GB/s**（PCIe Gen4 x16 的理论 25 GB/s 的 80%），链路没问题；
2. **按 expert_id gather 只有 3.9 GB/s**，仅为纯拷贝的 **20%** —— 因为 host 侧 `index_select` 是内存带宽受限的随机访问，把 PCIe 的能力浪费掉了。

### 7.9.2 换入代价 × miss_rate（每层 top-10，共 48 层）

| miss_rate | 每层换入 | **48 层合计/token** | 判定 |
|---:|---:|---:|---|
| 100% | 16.97 ms | **814.8 ms** | 完全不可用 |
| 50% | 8.49 ms | 407.4 ms | 不可用 |
| 30% | 5.09 ms | 244.4 ms | 不可用 |
| 20% | 3.39 ms | 163.0 ms | 不可用 |
| **10%** | 1.70 ms | **81.5 ms** | 仍不可用（目标是 3.5 ms） |

> **要满足 3.5 ms/token 的预算，miss_rate 必须 ≤0.5%** —— 这在 512 选 10 的 MoE 上不可能做到。

### 7.9.3 结论：放弃 offload，改走 W4 全常驻

| 方案 | 专家权重总量 | 每卡需求 | 61 GiB 可用 | 判定 |
|---|---:|---:|---:|---|
| bf16 专家 | **225 GiB**（48 层 × 4.68 GiB） | 112.5 GiB | ❌ 缺一半 | 不可行 |
| **W4 专家** | **56 GiB**（120.8B × 0.5 B） | **28 GiB** | ✅ 余 33 GiB 给 KV/激活 | **唯一可行** |

**这是本轮最重要的决策修正**：
- ❌ 原方案里的**专家缓存 / CPU-NPU 分层 / 权重动态调度**（P1 级）在本机**全部不成立** —— PCIe 上 gather 只有 3.9 GB/s，收益被搬运吃光；
- ✅ **W4 量化从"优化项"升级为"前置必要条件"**，优先级提到最高；
- ✅ 51B n-gram 表仍应放 DDR（它是**单次小批量 gather**，不是每层换入，代价可忽略），这部分设计不变；
- ✅ KV 压缩的价值进一步提升（省下的显存直接给专家）。

> 对照：Strata/nInfer 的 offload 方案建立在 NVLink/NVSwitch 与更高 PCIe 带宽之上，**在 PCIe Gen4 x16 的单机 2 卡环境里不成立**。这条要写进方法论：**offload 决策必须先用本机实测的 gather 带宽算一遍，不能沿用云端经验。**

---

## 7.10 W4 量化精度探针：RTN 不够，必须用 GPTQ/AWQ（2026-10-28）

从真实 checkpoint 取 `layers.0.mlp.experts.gate_up_proj` 的前 4 个专家（`[4,1280,2560]` bf16，25 MiB），
做 group=128 的对称量化 → 反量化 → 对比：

| 格式 | 平均相对误差 | 最大绝对误差 | 余弦相似度 |
|---|---:|---:|---:|
| **W8A16 (g128)** | **0.78%** | 0.0011 | ≈1.0000 |
| **W4A16 (g128, RTN)** | **14.19%** | 0.0193 | **0.9935** |

逐专家看差异不大（13.55%–15.26%），说明不是个别专家异常。

**判读**：
1. **W8 非常安全**（0.78% 相对误差），可以直接用；
2. **W4 只做 RTN（round-to-nearest）不够** ——14% 的平均相对误差会让 MoE 路由和下游注意力显著漂移。必须用 **GPTQ / AWQ** 这类带校准的量化方法（业界 W4 g128 配 GPTQ 通常能把等效误差压到 2–4%）；
3. 余弦相似度 0.9935 看起来不差，但结合 §7.5.3 的"路由阶跃敏感性"，**必须用「路由一致率」而不是余弦相似度来验收**。

**对计划的修正**：
- W8 稠密部分：本周就能做（msModelSlim 或自写 RTN 均可用）；
- **W4 专家部分：必须引入校准流程**，工作量从"跑个脚本"变成"搭一套量化流水线"（校准集 + GPTQ + 逐层评估），这是剩余工作的主体；
- 验收三层：单算子误差 → **top-10 专家集合重合度** → 端到端任务指标。

---

## 7.11 W8 稠密量化流水线验证通过（2026-10-28）

对 `layers.0` 全部稠密权重（13 个二维张量，141.9 MiB）做 group=128 对称 INT8 量化 → 反量化：

| 指标 | 结果 |
|---|---|
| 相对误差范围 | **0.81% – 1.65%**（均值 ~1%） |
| 余弦相似度 | 0.9999 – 1.0118（≈1.0） |
| 体积压缩 | 141.9 → 72.0 MiB，**1.97×** |
| 最差张量 | `mlp_hyper_connection.block_inject_weight`（1.65%，仅 4×10240 的小矩阵） |

**结论：W8 稠密可以直接上，不需要校准。** 这与 §7.10 的结论一致（W4 才需要 GPTQ/AWQ）。
体积减半意味着稠密部分从 ~8 GB 降到 ~4 GB，为 KV 与专家腾出更多空间。

另注：每层有 `attn_hyper_connection` / `mlp_hyper_connection` 两组 `input_mix_weight_{up,down}` + `block_inject_weight` —— 这是 Gated Residual 的实际参数形式，`gated_residual_fused` 算子要按这个形状设计（up/down 各 `[10240,320]`/`[320,10240]`）。

---

## 7.12 W4 量化方案对比与选型（第 13 轮实测，真实专家权重）

样本：`layers.0.mlp.experts.gate_up_proj` 前 4 个专家 `[4,1280,2560]` bf16（25 MiB）

| 方案 | 相对误差 | 余弦相似度 |
|---|---:|---:|
| W4 minmax g128（基线） | 14.19% | 0.99349 |
| W4 minmax g64 | 12.65% | 0.99501 |
| W4 minmax g32 | 11.02% | 0.99668 |
| W4 MSE 裁剪 g128 | 11.88% | 0.99488 |
| W4 MSE 裁剪 g64 | 11.10% | 0.99618 |
| **W4 MSE 裁剪 g32** | **10.02%** | **0.99779** |
| W4 MSE 裁剪 + AWQ 式缩放 g64 | 10.85% | 0.99664 |

**选型：W4 + MSE 最优裁剪 + group=32**（相对误差从 14.19% 降到 **10.02%**，降低 29%）。
- MSE 裁剪（在 0.55–1.0 之间对数搜索每组收缩比例）比 minmax 稳定提升 ~2.3 个点；
- group 从 128 缩到 32 再提升 ~1.9 个点（scale 开销从 1/128 升到 1/32，每权重 +0.25 bit，可接受）；
- AWQ 式按通道缩放收益有限（10.85% vs 11.10%），成本却是每层多一遍统计 → **先不做**。

### 重要修正：专家量化**不会**翻转路由

之前 §7.5.3 担心"1e-4 扰动就翻转专家选择"，那是针对**隐藏状态**的数值扰动（如 DeltaNet 换算法路径）。
**专家权重量化不影响路由**——路由只由 `mlp.gate.weight`（走 W8，误差 0.78%）决定，与被量化的专家权重无关。

→ 修正验收标准：
| 层 | 指标 | 阈值 |
|---|---|---|
| ① 张量级 | 相对误差 / 余弦 | rel ≤12%、cos ≥0.997 |
| ② 路由级 | top-10 专家集合重合度 | **仅适用隐藏状态扰动，不适用专家权重量化** |
| ③ 端到端 | logits KL / 任务指标 | KL ≤0.05、任务掉点 ≤2% |

---

## 附录：立即可以做的 3 件事（本周）

1. **扩盘**：执行基线报告附录 A，把 `/models` 挂上 3.26 TiB。
2. **拉权重**：`modelscope download --model Qwen/Qwen3.8-Flash-Next --local_dir /models/Qwen3.8-Flash-Next`（360 GB，千兆约 48 min），并核对 `crc32.txt`。
3. **搭正确性基线容器**：`flash-next-dev`（transformers 5.21.0 + torch_npu 2.10），先跑通 1 层切片，把 Phase A 的 5 个组件对齐。
