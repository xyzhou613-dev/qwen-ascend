# Qwen3.8-Flash-Next 第 32 轮实测记录

> 本轮目标：**提高推理速度**。
> 结果：TPOT **139.5 → 121.1 ms（−13.2%，7.2 → 8.3 tok/s）**，`mean_nll` 保持 **2.1756** 不变。
> 做法只有一个关键词：**装载期预反量化**（把"每 token 重做一遍"的 int8→bf16 反量化改成装载期一次做完）。

---

## 0. 一句话结论

| 项 | 前 | 后 | 变化 |
|---|---|---|---|
| TPOT（96 token 输出） | 139.5 ms | **121.1 ms** | **−13.2%** |
| 吞吐 | 7.2 tok/s | **8.3 tok/s** | +15% |
| TTFT（热态） | 0.16 s | **0.14 s** | −12% |
| `mean_nll`（200 token） | 2.1756 | **2.1756** | 0（逐位相同） |
| 配置 | LPC=46 | **LPC=45 + 三级预反量化** | — |
| 冷启动 | 30 s | 31 s | +1 s（前缀缓存提前构建） |

---

## 1. 为什么这个优化管用

第 30 轮 §16 量到：**每 token 的最大开销是"把整层 int8 权重重新反量化一遍"**。
`tests/t_batch_micro.py` 进一步给出关键事实：

| | 耗时 |
|---|---|
| int8 权重 `E.lin(x, W)`（含反量化） | **0.20 ms** |
| 同一块权重的纯 bf16 matmul | 0.04 ms |
| 同一个 int8 权重、B=1 / B=8 / B=32 | 0.20 / 0.19 / 0.19 ms |

⇒ 成本的 80% 是**反量化这一次固定动作**，与 batch、与矩阵形状几乎无关。
既然权重在显存里是常驻的，那就**在装载期反量化一次、之后直接 GEMM**：

- 数值上**逐位相同**（反量化本来就是 `q * scale`，只是不再每 token 重算）；
- 代价是显存：bf16 比 int8 大一倍。

## 2. 三级预反量化（三个 flag，可单独开关）

| flag | 对象 | 每层字节 | 实测收益 | 显存代价 |
|---|---|---|---|---|
| `FNX_DEQ_SMALL=1` | 每层 11 个小权重（`a_down/a_up/m_down/m_up/a_inj/m_inj/sh_gate/sh_up/sh_down/sh_gate2/gate_w`） | 19.8 MiB int8 → 39.6 MiB bf16 | **−15.6 ms（−10.8%）** | +10 MiB/层 |
| `FNX_DEQ_HEAD=1` | `lm_head`（606 MiB int8，原来分 16 块每 token 重反量化） | 606 MiB → 1.18 GiB | **−5.5 ms（−3.9%）** | +0.6 GiB（放 dev1，那里空） |
| `FNX_DEQ_ATTN=10` | 前 10 层的注意力投影（`w_qkv/w_z/w_out` 等） | 55 MiB int8 → 110 MiB bf16 | **−7.2 ms（每层 0.72 ms）** | +55 MiB/层 |

**最反直觉的一行是第一行**：11 个小权重只占稠密字节的 **26%**，却贡献了 **10.8% 的总加速** ——
因为它们单个都太小，反量化开销与权重尺寸无关，所以"越小越亏"。

## 3. 为什么最后是 LPC=45 而不是 46

预反量化要吃显存（小权重 +10 MiB/层、注意力 +55 MiB/层），而 dev0 在 LPC=46 时只剩 ~0.5 GiB。
于是有一个明确的取舍：

| 方案 | TPOT | 说明 |
|---|---|---|
| LPC=46 + 小权重 + head（无 attn） | OOM | 差 24 MB |
| LPC=46 + head（无小权重/attn） | 135.9 ms | 只吃 head 的收益 |
| **LPC=45 + 小权重 + head + attn=10** | **121.1 ms** | ✅ 定稿 |
| LPC=45 + 小权重 + head（attn=0） | 129.4 ms | |
| LPC=44 + 小权重 + head + attn=32 | 123.3 ms | 多丢一层捕获，反而不划算 |

**结论：少捕 1 层（代价 ~12 ms）换来 32 层份的预反量化预算，是这轮最赚的一笔账。**

## 4. 上限在哪：attn=10 之后就是悬崖

- `attn=12`：装配能过、前缀缓存也能建，但**图 replay 直接失败**：
  ```
  RuntimeError: replay ... error code 507899
  Resource_Error_Insufficient_Device_Memory(EL0019): Failed to allocate 33554432 bytes
  ... halStreamTaskFill failed ... Build SQ/CQ failed ... rtModelExecute execution failed
  ```
  图 replay 需要 ~32 MB **驱动侧**内存（SQ/CQ 绑定），torch 的缓存分配器帮不上忙。
- `attn=14`：装配阶段就 OOM（差 62 MB）。
- ⇒ **attn=10 是安全上限**，对应的 dev0 稳态占用 59.57 GiB（reserved 60.41，驱动可用余量 ~0.55 GiB）。

## 5. 顺带修掉的两个真 bug

### 5.1 前缀缓存构建失败会变成"每请求 500"

原先 `_build_prefix` 的异常只在 `Engine.__init__` 里被吞掉，但**没有记录失败**；
`_prefix_hit` 每个请求都会再试一次构建 → 显存贴边时**每个请求都 OOM → 全部 500**
（流式客户端看到的是"空回答"，因为 SSE 里一个 content 都没有）。
修：失败时写入 `([], None)` 哨兵，该模板变体**永久降级为无缓存**，服务继续可用。

### 5.2 前缀缓存应该在建图之前构建

前缀缓存构建要克隆 ~20 MB 状态；放在图捕获之后时，图的池子已经占住显存，贴边就 OOM。
把顺序改成 **装配 → 建前缀缓存（eager 跑 3 个 token）→ 图捕获**，构建成功率明显提高
（attn=12 就能建成；虽然它最终仍因 replay 内存不足而不可用）。
副作用：冷启动 +1 s（eager 跑 3 个 token）。

## 6. 复现

```bash
# 环境变量（其余同 baseline-r31）
FNX_LPC=45 FNX_NLAYERS=48 FNX_DEQ_SMALL=1 FNX_DEQ_HEAD=1 FNX_DEQ_ATTN=10
```

或用仓库脚本：`bash scripts/30_deploy.sh`（已包含这些 flag）。

## 7. 下一步还能做什么（按性价比）

1. **`w_z`/`w_out`/`w_qkv` 的剩余 35 层**：每层还有 ~0.72 ms 可挖（共 ~25 ms），
   但需要 ~1.9 GiB dev0 显存 —— 只能靠"再少捕 1–2 层"或把专家权重压到 3.5 bit 来换。
2. **MoE 专家权重（1.21 GiB/层，占 94%）** 目前是唯一没被动过的大块：
   top-10 专家数据 24.5 MB/层/token 的 gather + 解包。要提速只能靠**可用的 fused kernel**
   （第 29 轮的 kernel 路线精度问题未定位）。
3. **dev1 的 3 层 eager**：每层 ~12 ms（共 36 ms，占 121 ms 的 30%）。
   要收掉它们必须让 dev0 再腾出 3 × 1.29 GiB —— 与预反量化互相争夺同一块显存。
