# Qwen3.8-Flash-Next 第 37 轮实测记录

> 主题：**把专家 GEMM 从「每 token 解包 16.4M 元素 + 两次 bf16 GEMM」换成 aclnn 的 Cube 融合 per-group W4A16**。
> 产物：线上 tag `baseline-r37`。TPOT **110.3 → 76.7 ms**（13.0 tok/s），prefill **105 → 74.3 ms/token**，质量 `mean_nll 2.1756 → 2.1995`，MCQ **24/24** 不变。
> 本记录只写这一轮做过的事和量到的数；旧轮的推导见第 29–32 轮记录。

---

## 0. 一句话结论

**CANN 9.1 的 `npu_weight_quant_batchmatmul`（aclnnWeightQuantBatchMatmulV2）在 910B4 上支持 per-group W4A16。** 我们用「10 个专家沿 N 拼接成一次调用」+「dn 用 M=10 对角块」把它用在 MoE 上，一层专家段从 **1.62 ms 降到 0.32 ms**，整机 **−33.8 ms/token**。以前判它「不可用」是因为把 `161002` 的 dtype 校验错误读成了「不支持 per-group」。

---

## 1. 起点：文档里的两个前提都是错的

| 前提（第 25 轮记录 / 交接文档） | 实测复核结果 |
|---|---|
| `npu_weight_quant_batchmatmul` 只吃 per-channel | **错**。per-group 可用，条件是 `antiquant_scale` 的 dtype 必须与 `x` 相同（bf16/bf16） |
| `npu_grouped_matmul + antiquant` 报 161002 → 没有现成 per-group 算子 | **部分对**：grouped 版本确实不可用；但单个 matmul 可以用「拼 N」+「对角块」绕开 |
| 自研 AscendC kernel 是唯一出路（§4.10） | **错**：现成算子先达到了同样效果，且不用背 `s4t` 的 +5.6 GiB |

**错误原文**（现在能读到了）：

```
AclNN_Parameter_Error(EZ1001): x's dtype must be same as antiquantScale's dtype,
actual x's dtype is [DT_BFLOAT16], antiquantScale's dtype is [DT_FLOAT16].
```

`torch_npu` 自带 docstring 明确写着：*Atlas A2 训练系列产品：per-tensor、per-channel、**per-group***；int4 用 `npu_convert_weight_to_int4pack` 的 int32 容器承载；per-group 的 scale 形状要求 `(ceil(K, group_size), N)`；`antiquant_group_size ∈ [32, K-1]` 且必须是 32 的倍数；`inner_precise=1` 只在「int32 weight + NZ 格式 + M≤16 + per-group」时可用。

---

## 2. 语义与数值验收（自包含探针）

探针：`backend/remote/p319_wqbmm4.py`…`p326_capture.py`（都在仓库 `ops/scripts` 留档）。全部在容器 `fnx` 内跑，E=512 的真实分布（避免小 artifact 落进 L2 造成假快）。

| 项 | 结果 |
|---|---|
| int4pack 布局 + bf16 scale + `group_size=64` | **数值 PASS**：与 bf16 反量化+bmm 参考 rel **2.485e-3** |
| 真实权重（L00，与线上 `ExpQ` 逐专家对比） | gu **rel 2.17e-3**、dn **rel 1.63e-3** |
| 10 次单专家调用（gu） | 0.458 ms —— 太慢，必须拼接 |
| 10 专家沿 N 拼接成一次调用 | **0.043 ms**（K=2560 → N=12800） |
| 只拼接（gather+permute） | 0.069 ms（`[E,K,N/8]`）→ **0.036 ms**（改存 `[K,E,N/8]`，省掉 permute） |
| scale 拼接三档 | `[E,N,G]` int8 + base：0.213 ms；`[G,E,N]` int8 + base：0.123 ms；`[G,E,N]` bf16（已乘 base）：**0.033 ms**（但 +1.5 GiB，暂不采用） |
| dn：M=10 对角块法 vs 10 次单专家 | **0.054 ms vs 0.481 ms（8.9×）** |
| 一层专家段合计（图内） | **0.32 ms**（旧 torch 路径 **1.0 ms**） |
| NPUGraph 捕获 | **replay vs eager rel 0.000e+00**；换一组专家 ids 后仍 **0.000e+00**；连续 replay 输出有限 |

> 单层专家段 eager 口径另有一组对照（含权重同时加载、同一 token）：旧 1.62 ms → 新 0.52 ms（**3.1×**）。差别在于 eager 还含宿侧发射开销，图内才是真实设备成本。

---

## 3. 实现（改了什么）

1. `fnx/engine.py`
   - 新增 `ExpW4` 容器：`w4 [K,E,N/8] int32` + `s8t [G,E,N] int8` + `base [E,N] bf16`；
     `_wcat()`（`index_select(dim=1)` + `reshape`）、`_scat()`（3 个算子现场拼 scale）、`apply()`、`apply_diag()`。
   - `FnLayer._moe` 增加新的分支：gu 一次调用出 `[1, 10×1280]`，dn 走对角块，最后加权求和 + 共享专家。
   - 只支持 `FNX_BATCH=1`（`b>1` 显式抛错，避免静默错算）。
2. `fnx/load.py`：`FNX_MOE_W4A16=1` 且 `L??.w4.safetensors` 存在时，直接从新 artifact 构建 `ExpW4`，**不再加载旧的 v2 专家张量**（显存不翻倍）。
3. `tools/repack_w4a16.py`（新）：把 LQP 的 `v2o/s8/base` 转成 int4pack 布局。
   - 逆 v2 布局：`byte(64s+j)` 低 4 bit = `W[128s+j]`、高 4 bit = `W[128s+64+j]`，`value = nibble − 8`；
   - 逐专家 `npu_convert_weight_to_int4pack`，堆成 `[K,E,N/8]`；
   - 48 层 × 2 投影，**约 22 分钟**（每层 27 s），产出 57.7 GiB，与原 artifact 同大小。
4. `tests/t_w4a16_check.py`（新）：在真实 L00 权重上对比 `ExpQ` 与 `ExpW4`，**必须先设 `FNX_PACKED_TORCH=1 FNX_EXP_G=64 FNX_LQP=...`**，否则比的是 g32 的原始 Q 权重（第一次就踩了这个坑，rel 7.2e-2 的假失败）。

---

## 4. 端到端 A/B（同一脚本、同一天、同一台机器）

脚本：`backend/remote/p327_w4a16.sh`（`W4=0/1` 只改一个环境变量）。

| 口径 | W4=0（r35 基线） | W4=1（r37） | 差 |
|---|---:|---:|---|
| TPOT（128 token，两轮） | 114.5 / **110.5** ms | 86.5 / **77.4** ms，复测 79.4 / **76.7** | **−33.8 ms（−30.6%）** |
| 吞吐 | 9.0 tok/s | **13.0 tok/s** | **+43%** |
| prefill | ≈105 ms/token（历史口径） | **74.3 ms/token**（231 token → 17.17 s） | −29% |
| TTFT 热态 | 0.13 s | **0.10 s** | — |
| TTFT 冷（9/77/231 token） | — | **0.79 / 5.87 / 17.17 s** | 248 token 口径从 29.3 s → 17.2 s |
| `mean_nll` | **2.1756** | **2.1995** | +0.024 nats |
| `ppl` | 8.81 | 9.02 | — |
| top1 | 49.0% | 48.0% | −1 点 |
| MCQ（24 题） | 24/24 | **24/24** | 持平；4.16 s/题 |
| dev0 显存 | 59.57 GiB | **59.57 GiB** | 持平 |

**质量差异的归因**：两套路径的权重与 scale **逐位相同**，差别只在 GEMM 的累加方式——旧路径是两次 bf16 GEMM（各 1280 深）后相加，新路径是一次 fp32 深度累加（K=2560）。这是数值重排，不是量化变差；任务级指标未退化（MCQ 24/24）。

---

## 5. 时间账（r37 之后 76.7 ms 花在哪）

| 段 | 值 | 依据 |
|---|---:|---|
| 整机 TPOT | **76.7 ms** | 同脚本 A/B 实测 |
| 其中：专家段（45 层合计） | **≈7 ms**（旧 ≈45 ms） | 探针：图内 0.32 ms/层 vs 旧 1.0 ms/层 |
| 其中：dev1 3 层 eager | **≈23.6 ms** | 7.85 ms/层（第 33 轮修复后） |
| 其中：head（dev1） | **≈5.7 ms** | 第 33 轮 |
| 其余（dev0 的稠密/attention/DeltaNet + 宿侧空隙） | **≈40 ms**【推算】 | 76.7 − 7 − 23.6 − 5.7 |

> 口径：最后一行是**减出来的**，里面混着「dev0 图内真实设备时间」和「宿侧 15–17 ms 空隙」，**不能当单一指标用**。r37 的算子结构已变（Mul/Cast/Subs 那一大簇消失），宿侧占比会上升，**下一轮第一件事是用 `FNX_PROF=1` 重采一次 msProf**。在拿到新 profile 前，只有两条是硬结论：**专家段从 45 ms 量级掉到 7 ms 量级**（探针直接测）、**整机 −33.8 ms**（同脚本 A/B 直接测）。

---

## 6. 踩到的坑（按付出代价排序）

1. **`nan` 的「图捕获失败」是假的**：第一次图捕获测试用 `y = fn()` 读图池里的输出，得到 `nan`；真实原因是捕获期的输出张量不能直接读。改成「捕获前分配常驻 buffer、图内 `copy_` 进去」后 **rel 0.000e+00**。生产代码本来就是这么做的（`body0` 最后 `self.h0.copy_(h)`），是探针写错了。
2. **scale 必须同 dtype**：见 §1。这一条让整条路被封了 12 轮。
3. **scale 张量必须是连续或「最后两维转置」**：`permute(...).reshape(...)` 可能返回一个 stride 是 `(1, G)` 的转置视图，acl 会报 `only support antiquantScale tensor is contiguous or transpose last two dims`；`.contiguous()` 一下就好。
4. **拼接后的 scale 形状**：`[G, k*N]`，`k*N` 的顺序必须与权重拼接顺序一致（都是专家主序），否则数值错但**不报错**。
5. **测试口径必须先对齐生产**：用 g32 的原始 Q artifact 去比 g64 的 LQP 结果，得到 rel 7.2e-2 的假失败。
6. **流式客户端提前断开会让服务端卡死**（`/health` 阻塞、NPU 空闲、`SIGTERM` 无效，只能 `kill -9`）。我的 TTFT 脚本在首 token 处 `break` 就触发了。生产侧这是真实的稳定性风险，建议下一轮加「请求取消时强制释放 LOCK + 超时兜底」。

---

## 7. 下一轮的建议顺序

1. 用 `FNX_PROF=1` 重采 msProf，确认 76.7 ms 的新分布（尤其宿侧空隙是否变大）。
2. **稠密权重也走 aclnn W8 per-group**：`w_qkv/w_z/w_out/w_q/w_k/w_v/w_o` 现在还是「每 token 反量化 → bf16 GEMM」；同时可释放 `FNX_DEQ_ATTN` 的 0.55 GiB 预反量化副本。
3. **dev1 的 3 层 eager（≈23.6 ms，占 31%）**：图捕获已证否，只能在算子数上做（本轮已让它们的 MoE 从 1.62 → 0.52 ms/层）。
4. 服务端取消/超时的兜底（§6.6）。
5. chunked prefill：ROI 已从 2.6–3× 降到 1.6–1.8×，排在上面之后。

---

## 8. 复现清单

```bash
# 1) 生成新 artifact（一次，约 22 分钟）
docker exec fnx bash -lc 'cd /root/ascend-dev/fnx && \
  /root/ascend-dev/venv/bin/python tools/repack_w4a16.py 48'

# 2) 一致性门禁（真实 L00 权重，新旧容器对比）
docker exec fnx bash -lc 'cd /root/ascend-dev/fnx && \
  FNX_PACKED_TORCH=1 FNX_EXP_G=64 FNX_LQP=/models/Qwen3.8-Flash-Next-LQP \
  /root/ascend-dev/venv/bin/python tests/t_w4a16_check.py'
# 期望：gu rel ≈2.2e-3、dn rel ≈1.6e-3、一层专家段 3.1×

# 3) 起服务 + 质量 + TPOT（W4=0 即 A/B 的基线）
W4=1 bash /root/ascend-dev/p327_w4a16.sh
# 期望：mean_nll 2.1995 / TPOT 76–80 ms / hbm0 59.57 GiB

# 4) 任务级
docker exec fnx bash -lc 'cd /root/ascend-dev/fnx && \
  /root/ascend-dev/venv/bin/python tools/mcq_eval.py'
# 期望：24/24
```
