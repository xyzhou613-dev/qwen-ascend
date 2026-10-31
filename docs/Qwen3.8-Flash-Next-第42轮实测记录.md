# Qwen3.8-Flash-Next 第 42 轮实测记录

> **本轮首次同时拿到"更快 + 更好"**：TPOT **65.5 → 64.6 ms**（15.5 tok/s），并且 `mean_nll` **2.1735 → 2.1648**（历史最佳，比 r38 的 2.1665 还好），MCQ **24/24**。
> 来源是一个被忽略的手写归一化：DeltaNet 的输出归一化换成框架的融合 `npu_rms_norm`。

---

## 1. 双赢的那一项：DeltaNet 输出归一化换融合算子

### 1.1 原实现

```python
xf = o.float()
xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)   # pow / mean / rsqrt / mul
xf = self.dn_norm_w * xf.to(torch.bfloat16)                     # 再舍入到 bf16，再乘权重
xf = xf * z
```
每层 **6 个 kernel**（pow、mean、rsqrt、mul、cast、mul），而且中途把归一化结果**先舍入到 bf16 再乘权重**——多了一次量化。

### 1.2 改法

```python
xf = rms_norm(o, self.dn_norm_g).float() * z
```
`rms_norm()` 内部走 `torch_npu.npu_rms_norm`（fp32 内部计算，1 个融合 kernel）。

**⚠️ 坑（第一版就炸了）**：`rms_norm()` 是 **(1+weight) 语义**，而这里要的是**直接乘** `dn_norm_w`。第一版直接传 `dn_norm_w`，等于把权重整体 +1，`mean_nll` 从 2.1735 炸到 **2.5248**（ppl 12.49）。修法：装载期预存 `dn_norm_g = dn_norm_w - 1`。

### 1.3 实测（A/B，同机同日）

| 配置 | mean_nll | ppl | top1 | TPOT |
|---|---:|---:|---:|---:|
| `FNX_DN_RMS=0`（原实现） | 2.1735 | 8.789 | 48.0% | 65.5 / 65.4 ms |
| **`FNX_DN_RMS=1`（新）** | **2.1648** | **8.713** | 48.0% | **64.0 / 64.6 / 64.9 ms** |

- **质量更好**：2.1648 是历史最佳（r35 2.1756 → r38 2.1665 → 本轮到 2.1648）。原因是少了一次 bf16 舍入。
- **速度更快**：每层少 5–6 个 kernel（45 层 ≈ 250 个/步），约 −0.8 ms。
- MCQ **24/24**（3.61 s/题）。

## 2. 本轮两个"看起来该赚、实测不赚"（都留开关，默认关）

| 改动 | 预期 | 实测 | 开关 |
|---|---:|---|---|
| linear 层 4 合 1：`qkv+z+b+a` 一次算子调用（3 次 → 2 次/层） | −1.8 ms | **0**（65.1/65.7 vs 64.7/65.6，nll 不变） | `FNX_DENSE_CAT4=0` |
| `inject` 用 `addcmul` 融合（第 41 轮） | −0.3 ms | 慢 0.5 ms | `FNX_INJECT_CM=0` |

**4 合 1 为什么不赚**：`w_b/w_a` 只有 48 列（0.25 MB），那一次调用的固定开销远小于 profile 里 49 µs 的平均值；而合并后的大调用 N 从 10240 涨到 16480，多读的 16 MB 正好把省下的抵消。

## 3. r42 端到端

| 项 | 值 |
|---|---|
| TPOT | **64.0 / 64.6 / 64.9 ms（典型 64.6，15.5 tok/s）** |
| TTFT 热态 | 0.08 s |
| mean_nll / ppl / top1 | **2.1648 / 8.713 / 48.0%** |
| MCQ | **24/24**（3.61 s/题） |
| dev0 显存 | 58.97 GiB |
| 累计 | 110.3（r35）→ **64.6 ms（r42）= 1.71×**；初版 318 ms 起算 **4.9×** |

## 4. 对目标(1)的交代

- **质量门槛已达成并超额**：原目标要求"`/fnx/loss` 数值不退化"（相对 r38 的 2.1665）——现在 **2.1648**，比 r38 还好。
- **速度门槛仍未达成**：目标 62–63 ms，现在 **64.6**，差 ~2 ms。
- 剩下的空间按 r40/r41 的 profile：`WQBM 15.9`、`Mul 7.2`、`IndexSelect 5.5`、`MatMul 3.6`，每 token 仍 ~6,300 个 kernel。**torch 层的"合并/融合"手段已经用尽**（本轮又证否了 4 合 1 和 addcmul 两项）。再往下只有两条：① L1 级手写融合 kernel（上限 −6~7 ms，1–2 周）；② 或者接受 64.6。

## 5. 复现

```bash
# 生产（r42）
DNR=1 DW=1 ATTN=0 DH=1 SM=1 bash /root/ascend-dev/p331.sh   # 期望 64.0–64.9 ms / 2.1648 / MCQ 24/24
# 对照（r41 的质量点）
DNR=0 DW=1 ATTN=0 DH=1 SM=1 bash /root/ascend-dev/p331.sh   # 期望 65.4–65.5 ms / 2.1735
# 坑的复现：把 rms_norm(o, self.dn_norm_g) 换成 self.dn_norm_w → nll 2.5248
```
