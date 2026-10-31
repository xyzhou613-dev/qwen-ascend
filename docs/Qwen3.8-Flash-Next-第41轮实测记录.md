# Qwen3.8-Flash-Next 第 41 轮实测记录

> 本轮不安新东西，只做两件"决定要不要继续投"的事：
> **① 在 r40 配置下重采 msProf**（给出 65.5 ms 现在的真实分布）；
> **② 定量回答"融合 kernel 到底能赚多少、要付什么代价"** —— 结论比预期悲观：**把解包写到 GM 再乘，实测反而更慢**。

---

## 1. r40 配置下的设备时间分布（msProf，captured）

| 项 | 值 |
|---|---|
| 设备总时间 | **50.4 ms/步**（r38 是 53.4，r37 是 62.3，r34 是 88.5） |
| kernel 数 | 18,948 / 3 步 ≈ **6,300 个/步**（平均 8.0 µs，**没变**——减少的是"贵 kernel"的调用吗？不是，见下） |
| 墙钟 | 64.7–65.9 ms ⇒ 宿侧空隙 ≈ 15 ms |

| 算子 | ms/步 | 占比 | 相对 r38 |
|---|---:|---:|---|
| `WeightQuantBatchmatmul`（专家+稠密） | **15.9** | **31.6%** | 16.0 → 15.9（几乎没动） |
| `Mul` | 7.2 | 14.3% | 7.2 → 7.2 |
| `IndexSelect` | **5.5** | 10.9% | 5.3 → 5.5 |
| `MatMul` | **3.6** | 7.2% | 6.9 → **3.6（合并生效）** |
| `ReduceSum` / `Cast` / `Transpose` / `Mean` / `Slice` | 2.3 / 2.1 / 2.0 / 1.6 / 1.5 | — | 基本持平 |
| `Add` / `RepeatInterleave` / `RmsNorm` / `Sigmoid` | 0.9 / 0.9 / 0.6 / 0.5 | — | — |

**读法**：
- 本轮的合并（qkv/z、q/k/v、mixer、sh/gate）确实把 `MatMul` 从 6.9 压到 **3.6 ms/步**，但**总量只减 3 ms** —— 因为大头 `WeightQuantBatchmatmul`（15.9）和 `Mul`（7.2）没动。
- 每 token 仍有 **6,300 个 kernel**。要再往下，只能减少 kernel 数（融合），没有第二种办法。

## 2. 融合 kernel 的收益上限：先算再动手

### 2.1 MoE 段现在到底占多少

按本机实测（图内，`ops/script` 的 r37/r39 探针）：

| 组成 | ms/层 | 45 层 |
|---|---:|---:|
| 权重 gather（`index_select`，`[K,E,N/8]`） | 0.036 | 1.6 |
| scale 拼接（`[G,E,N]` int8 → bf16） | 0.033 | 1.5 |
| WQBM gu（N=12800，M=1） | 0.043 | 1.9 |
| WQBM dn（N=25600，M=10，对角块） | 0.054 | 2.4 |
| act / 对角取块 / 加权求和 / 共享专家 | ~0.15 | ~6.8 |
| **合计** | **≈0.32** | **≈14** |

⇒ 一个"理想融合 kernel"（gather + 解包 + GEMM 全在一个 kernel 里）的理论上限是**把这 14 ms 压到 3–4 ms，即省 6–7 ms（65.5 → ~58.5）**。这是这条路上唯一还值钱的一块。

### 2.2 但"把解包结果写到 GM 再乘"这条最省事的路，实测是负收益

融合 kernel 有两种写法，代价差一个量级：

| 写法 | 单层额外数据量 | 实测/推算代价 | 与现状比 |
|---|---|---:|---|
| (a) dequant 写到 **GM** 中转 buffer，再 `Matmul` 读 | 写 65.5 MB + 读 65.5 MB | **0.070 ms/层**（实测 D2D：65.5 MB 拷贝 0.070 ms = 1872 GB/s） | **比现在整个 gu 调用（0.043 ms）还慢** |
| (b) dequant 留在 **L1/UB**，Cube 直接从 L1 取 | 0（不出芯片） | 理论上 ≈ 现有 WQBM | 才有机会赚 |

> 这就是为什么"写个融合 kernel 就能快"是错的：**WQBM 已经把解包和 GEMM 融在一个 kernel 里了**，它的问题不是没融合，而是 M=1 形状下 49 µs/次的固定开销。自定义 kernel 只有做到 **(b) L1 级融合** 才有意义，而那是 MIX + 手工 L1 管理，正是 1–2 周的那部分。

### 2.3 因此对"要不要投"的判断

- **值得**：MoE 段 14 ms 里能省的 ~6–7 ms 是当前唯一 >1 ms 的单项空间；且 Cube 通道已证明可用（24.9 TFLOPS、多核正确）。
- **前提**：必须做 (b) 不是 (a)。(a) 已实测否决。
- **风险**：MIX 下手工管理 L1 + 4-bit 解包写 L1，是本项目至今最硬的一段；失败的话退回现状（无损失，因为都是旁路）。

## 3. 本轮两个小实验（都回退）

| 改动 | 预期 | 实测 | 处理 |
|---|---:|---|---|
| `inject` 的 `mul→reshape→add` 换成一次 `addcmul`（数值等价，每层省 2 个 kernel） | −0.3 ms | **慢约 0.5 ms**（64.8/65.9 → 66.0/70.7 首样本） | 回退，开关 `FNX_INJECT_CM` 默认 0 |
| （沿用第 40 轮结论）DeltaNet 去 `repeat_interleave` | −0.9 ms | 慢 0.7 ms | 已回退 |

> **这是第 34 轮那条结论的第二次复现**：这台机器上 `addcmul` 的广播实现比"乘+加两个原语"更慢，与张量大小无关（本次只有 10240 个元素）。以后不要再试。

## 4. r41 状态（与 r40 同配置，仅代码注释/开关变化）

| 项 | 值 |
|---|---|
| TPOT | **64.7 / 65.6 ms（15.3 tok/s）** |
| mean_nll / ppl / top1 | 2.1735 / 8.789 / 48.0% |
| MCQ | **24/24**（r40 同配置实测） |
| dev0 显存 | 58.97 GiB |
| systemd / watchdog | active / 运行中 |

## 5. 复现

```bash
# 分布
docker exec fnx bash -lc 'bash /root/ascend-dev/p330.sh > /tmp/prof.log 2>&1'   # 服务需先停
# GM 中转代价（融合 (a) 的否决依据）
docker exec fnx bash -lc '/root/ascend-dev/venv/bin/python - <<PY
import torch, time
a=torch.empty(2560,12800,dtype=torch.bfloat16,device="npu:0"); b=torch.empty_like(a)
for _ in range(5): b.copy_(a)
torch.npu.synchronize(); t=time.perf_counter()
for _ in range(30): b.copy_(a)
torch.npu.synchronize(); print("copy 65.5MB = %.4f ms" % ((time.perf_counter()-t)/30*1e3))
PY'
# 期望：≈0.070 ms（= 1872 GB/s），而现有 gu 调用是 0.043 ms
```
