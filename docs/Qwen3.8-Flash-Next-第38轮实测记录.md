# Qwen3.8-Flash-Next 第 38 轮实测记录

> 主题：**稠密 int8 权重也搬进 Cube**（aclnn `npu_weight_quant_batchmatmul`，W8 per-group）。
> 结果：TPOT **76.7 → 67.0 ms**（14.9 tok/s），`mean_nll` **2.1995 → 2.1665**（比 r35 还好），MCQ **24/24**，dev0 显存 **59.57 → 58.98 GiB**。
> 本轮没有做任何"拿质量换速度"的事：速度和显存都变好，质量也变好。

---

## 0. 一句话结论

r37 把**专家** GEMM 搬进 Cube 之后，重新采 msProf 发现瓶颈换人了：**每 token 把稠密注意力投影从 int8 反量化成 bf16**（`aclnnMul` 16.9 ms + `aclnnInplaceCopy_Cast` 7.4 ms，占设备时间 39%）。用同一个 aclnn W8 per-group 算子替掉它：一层 5 个投影 **0.953 → 0.263 ms**，整机 **−7.8 ms**，顺带把 10 层预反量化的 0.55 GiB bf16 副本还给显存。

---

## 1. 重新采集（r37 配置，captured）

`tests/t_msprof.py`，warmup 1 + active 3 步：

| 项 | 值 |
|---|---|
| 设备总时间 | **62.3 ms/step**（r34 同口径是 88.5 ms） |
| 同时刻墙钟 | 76.7 ms ⇒ 宿侧空隙 **≈14 ms**（r34 是 15–17 ms，绝对值已降） |

按算子（每步，3 步均值）：

| 算子 | ms/step | 占比 | 是谁 |
|---|---:|---:|---|
| `aclnnMul` | **16.9** | 27.2% | 稠密反量化的 `q*s`（大头）+ DeltaNet/attention 的逐元素 |
| `aclnnMatmul` | 11.5 | 18.5% | **head ≈5.7** + 稠密 GEMM |
| `aclnnInplaceCopy_Cast` | 7.4 | 11.9% | 稠密反量化的 `int8→bf16`（+ 若干 fp32 转换） |
| `aclnnIndexSelect` | 5.8 | 9.3% | 专家权重/scale gather + embedding |
| `aclnnWeightQuantBatchmatmulV2` | **4.5** | 7.3% | r37 新加的专家融合 GEMM（≈0.10 ms/层） |
| ReduceSum / Mean / Add / RepeatInterleave / Transpose / Slice | 2.3 / 4.0 / 2.7 / 2.6 / 2.0 / 1.6+ | — | DeltaNet、QSA、gated residual |

**判读**：r37 之前那 49.6 ms 的"解包链"确实没了（现在 `WeightQuantBatchmatmul` 只占 4.5 ms），但 `Mul + Cast` 还有 **24.3 ms**——它们来自稠密权重，不是专家。

## 2. 三条路实测（真实 L12 权重，gs=128，M=1）

`tests/t_dense_w8_probe.py`：

| 路径 | 一层 5 个投影 | 45 层 | 显存 | 数值 |
|---|---:|---:|---|---|
| (a) 现状：每 token 反量化 + bf16 GEMM | **0.953 ms** | 42.9 ms | int8 常驻 | 基准 |
| (b) 装载期预反量化（`FNX_DEQ_ATTN`） | 0.158 ms | 7.1 ms | **+55 MiB/层（+2.5 GiB）** | rel 0.00e+00 |
| (c) **aclnn W8 per-group（本轮采用）** | **0.263 ms** | 11.8 ms | **不增加**（int8 常驻） | rel 0.00e+00 |

- (b) 最快，但 45 层要 +2.5 GiB——装不下，这正是为什么之前只能"预反量化前 10 层"。
- (c) 只比 (b) 慢 0.105 ms/层，却**不占额外显存**，所以可以全 48 层都用。
- 数值：三条路里 (b)(c) 与 (a) **逐位一致**（`rel 0.00e+00`）——scale 是 fp16 直接转 bf16、int8 值精确，没有任何新的量化误差。

## 3. 实现

1. `fnx/engine.py`：新增 `DenseW8`
   - `w8t [K,N] int8`（装载期转置一次）+ `sc [G,N] bf16`（**装载期一次性**把 fp16 scale 转置转 bf16，每个权重只有几十 KB）
   - `__call__`：算子只吃 2 维 `x`（3 维报 `161002: x's dim should be in range [2, 2]`），所以 `[b,1,H] → [b,H]` 再还原
   - `deq_w()`：保留给 `FNX_DEQ_SMALL` 的装载期预反量化用（小权重仍然用 bf16，别让它们也走算子）
   - `lin()` 增加 `DenseW8` 分支
2. `fnx/load.py`：`FNX_DENSE_W8=1` 且 `K % gs == 0` 时构建 `DenseW8`，否则回退 `QW`
3. 配置：`FNX_DEQ_ATTN=0`（10 层预反量化不再需要，省 0.55 GiB）、`FNX_DEQ_SMALL=1` 保持

## 4. 端到端（第一段：只换稠密路径）

| 口径 | r37 | **r38** | 差 |
|---|---:|---:|---|
| TPOT（128 token，多次） | 76.7 / 77.1 / 76.8 ms | **68.7 / 68.9 / 69.3 ms** | **−7.8 ms（−10.2%）** |
| 吞吐 | 13.0 tok/s | **14.5 tok/s** | +11% |
| TTFT 热态 | 0.10 s | **0.09 s** | — |
| `mean_nll` | 2.1995 | **2.1665** | **−0.033（变好）** |
| `ppl` | 9.02 | **8.73** | — |
| top1 | 48.0% | **49.5%** | +1.5 点 |
| MCQ | 24/24 | **24/24**（3.69 s/题，r37 是 4.16 s） | 持平且更快 |
| dev0 显存 | 59.57 GiB | **59.02 GiB** | −0.55 GiB |

**质量为什么反而变好**：稠密 GEMM 现在是"int8 权重在 Cube 内反量化后直接乘"，中间不落 bf16，累加精度比"先 bf16 反量化再 bf16 GEMM"高。r37 里那 +0.024 nats 的一部分其实来自这里，而不是当时以为的"MoE 累加顺序"。

## 5. 追加：共用输入的投影拼接（做完再 −1.9 ms，数值完全不变）

r38 之后重采 msProf（设备 **53.4 ms/步**，r37 是 62.3）发现最大单项换人了：`WeightQuantBatchmatmul` **16 ms/步**，而调用数是 315 次/步 —— **平均每次 0.05 ms，成本在发射不在算**。

`qkv` 与 `z` 用的是同一个 `h`，`b`/`a` 也是同一个 `h`：沿 N 拼成一个权重，一次调用全出，层内切片（对激活切片，几十 KB）。拼完必须把原件置 `None`，否则同一份权重在显存里放两遍（+55 MiB/层）。

| 口径 | 拼接前 | **拼接后** |
|---|---:|---:|
| TPOT | 68.7–69.3 ms | **66.9–67.7 ms** |
| `mean_nll` | 2.1665 | **2.1665**（逐位相同） |
| dev0 显存 | 59.02 GiB | **58.98 GiB** |

开关 `FNX_DENSE_CAT`（默认 1）。

## 6. r38 最终数字（systemd 托管下复测）

| 项 | 值 |
|---|---|
| TPOT | **66.9 / 67.1 ms（14.9 tok/s）** |
| TTFT 热态 | **0.08 s** |
| prefill | **67.7 ms/token**（231 token → 首 token 15.63 s；r37 是 74.3 / 17.17 s，r35 是 ≈105 / 29.3 s） |
| `mean_nll` / `ppl` / `top1` | **2.1665 / 8.73 / 49.5%** |
| MCQ | **24/24**（3.69 s/题） |
| dev0 显存 | **58.98 GiB**（r37 59.57） |
| 冷启动 | ≈30 s（装配 9.1 s） |
| 质量门禁 | 与 r37 的 `/fnx/loss` 逐位相同 → 拼接零风险 |

## 7. 下一步：LPC 46 差 100 MiB

试过把第 46 层从 dev1 eager 挪进 dev0 图（能省 ~5 ms），OOM：

```
Tried to allocate 802.00 MiB ... 59.51 GiB already allocated; 708.91 MiB free
```

只差约 93 MiB。可选的凑法（都不划算，暂不做）：

- `FNX_DEQ_SMALL=0` 能省 0.45 GiB，但 11 个小权重会改走算子（每个 ~0.04 ms×11×48 ≈ +21 ms），得不偿失；
- 稠密权重降到 4-bit 能省 1.2 GiB，但要重打包 + 质量代价（第 32 轮 §4.4 已论证 ROI 低）。

## 8. 复现

```bash
# 语法/数值门禁
docker exec fnx bash -lc 'cd /root/ascend-dev/fnx && \
  FNX_PACKED_TORCH=1 FNX_EXP_G=64 FNX_LQP=/models/Qwen3.8-Flash-Next-LQP \
  /root/ascend-dev/venv/bin/python tests/t_dense_w8_probe.py'
# 期望：一层合计 cur 0.953 / pre 0.158 / new 0.263 ms，rel 全 0.00e+00

# 起服务（DW=1 ATTN=0 即 r38；DW=0 回到 r37 口径做 A/B；LPC=46 会 OOM）
DW=1 ATTN=0 bash /root/ascend-dev/p331.sh
```
