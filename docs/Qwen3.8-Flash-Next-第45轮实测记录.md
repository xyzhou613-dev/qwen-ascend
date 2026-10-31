# Qwen3.8-Flash-Next 第 45 轮实测记录

> 主题：**prefill 还能快多少**。先用真实权重把"分块摊薄系数"测出来，再给目标数字。
> 现状：prefill **61–71 ms/token = 14–16 token/s**（r44 实测）。这确实不可接受——它和 decode（15.5 tok/s）几乎一样快，说明 prefill 完全没享受到"一次读一遍权重"的红利。

---

## 1. 实测：分块摊薄系数（真实 L00 权重）

一口算 T 个 token 时，各块的每 token 成本随 T 的变化（eager，绝对值偏大，**看倍数**）：

| 块 | T=1 | T=8 | T=32 | 摊薄倍数（T=32） |
|---|---:|---:|---:|---:|
| **稠密 GEMM**（一层 16 个稠密权重各一次） | 1.312 ms/tok | 0.158 | **0.040** | **32.7×** |
| **MoE 权重**（固定 10 专家，一次 WQBM） | 0.266 ms/tok | 0.032 | **0.009** | **31.1×** |

⇒ **纯权重读取这一层，T=32 能摊到 1/30。** 这与第 36 轮的稠密 31.9× 一致。

## 2. 但 MoE 有个结构性障碍：每个 token 的专家不同

T=32 的 chunk 里，专家的**并集**约 **238/512**（去重后）。当前 MoE 的实现是"10 个专家沿 N 拼成一次调用"，把它推广到 chunk 有两种写法，都不理想：

| 写法 | 每 token 权重读 | 每 token 算力 | 问题 |
|---|---:|---:|---|
| 并集沿 N 拼接 + 取对角块 | 18 MB（比现在 22 MB 还少） | **3.5 ms**（24× 算力浪费） | 浪费来自 T×U 块的笛卡尔积 |
| 按专家分组（grouped GEMM） | 18 MB | ~0.6 ms | 需要 routing 重排 + 分组 matmul（`npu_moe_init_routing` 有，但 `npu_grouped_matmul` 的 antiquant 在 CANN 9.1 报 161002，只能先 dequant 到 GM 再用 bf16 grouped matmul；chunk 规模下 GM 往返摊薄后只要 ~0.02 ms/token，可接受） |

## 3. 目标推算

`每 token 成本 = 稠密/T + MoE(并集) + 逐元素/递推/attention（不摊薄） + 宿侧空隙/T`

| 项 | 现在（ms/token） | 分块后（T=32） | 依据 |
|---|---:|---:|---|
| 稠密 GEMM | ~12.7 | **~0.4** | 实测 32.7× |
| MoE | ~7 | **1–4** | 权重实测 31×；并集写法浪费 24× → 3.5 |
| 专家/scale gather | 5.6 | ~4 | 并集 24/32 = 0.75× |
| 逐元素 + ΔNet 递推 + attention | ~19 | 10–19 | **真实计算不随 T 摊薄**，只有发射开销摊薄 |
| 宿侧空隙 | ~17 | ~1 | 1/T |
| **合计** | **61–63** | **≈16–28** | |

⇒ **30–60 token/s，取中位 40–45 token/s（2.5–3×）**，这是"纯 torch 代码、不改数值语义"能拿到的。

### 再往上需要两件事

| 步骤 | 收益 | 代价 |
|---|---:|---|
| MoE 改**按专家分组**（去重后 grouped matmul，去掉 24× 算力浪费） | MoE 4 → ~1 ms/token | 2–3 天（routing 重排 + GM dequant + grouped bf16 matmul + 数值重验） |
| 逐元素的 19 ms 用**融合算子/torchair**压 | 19 → 8–10 | 需先修 torchair（本机 import 报 protobuf 元数据缺失）；或手写融合算子 |
| 两者都做完 | **~50–100 token/s** | 合计 1–2 周 |

> 内存地板核算：T=32 的 chunk 每 token 必读 ≈ 3.4 GB/32 = 106 MB → 在 1447 GB/s 下 **0.073 ms/token（13,700 tok/s）**。所以 prefill 离硬件极限还有两个数量级，**瓶颈全在"每 token 的逐元素 + 发射"，不在带宽**。

## 4. "需要多少 tok/s"取决于你能接受的首 token 延迟

| prompt | 现在 15 tok/s | 45 tok/s（分块） | 100 tok/s（分块+MoE分组+融合） | 1000 tok/s（Strata 级） |
|---:|---:|---:|---:|---:|
| 2,048 token | 2.2 min | 45 s | 20 s | 2 s |
| 8,192 token | 8.8 min | 3.0 min | 82 s | 8 s |
| 16,384 token | 18.9 min | 6.1 min | 164 s | 16 s |

- 若目标"8k prompt 在 3 分钟内出首 token" ⇒ **≥45 tok/s**（分块 prefill 即可）
- 若目标"8k prompt 在 1 分钟内" ⇒ **≥135 tok/s**（分块 + MoE 分组 + 逐元素融合都要做）
- 若目标"和 Strata 的 1000 tok/s 一个量级" ⇒ 必须走**融合 kernel**（把解包/逐元素都融进 Cube），那是多周工程

**一个健康的分工参考**：decode 已经 15.5 tok/s，prefill 至少应是它的 5–10 倍（75–150 tok/s）才算"读 prompt 不卡"。所以现实目标建议定在 **80–100 tok/s**（8k prompt ≈ 80–100 s），而不是先追 Strata 的 1000。

## 5. 生产状态

测量用的脚本 `tests/t_prefill_chunk.py` 是只读探针（不改生产）；服务已恢复：`FNX_CTX=2048`、systemd `active`、dev0 58.97 GiB、`/health` ok、TPOT 64.6–65.1 ms。

## 6. 复现

```bash
# 分块摊薄系数（服务要先停）
docker exec fnx bash -lc 'cd /root/ascend-dev/fnx && \
  FNX_PACKED_TORCH=1 FNX_EXP_G=64 FNX_LQP=/models/Qwen3.8-Flash-Next-LQP \
  FNX_MOE_W4A16=1 FNX_W4A16_DIR=/models/Qwen3.8-Flash-Next-LQP-w4a16 FNX_DENSE_W8=1 \
  FNX_DEQ_ATTN=0 FNX_SMALL_MERGE=1 FNX_DN_RMS=1 \
  /root/ascend-dev/venv/bin/python tests/t_prefill_chunk.py'
# 期望：稠密 T=1 1.312 → T=32 0.040 ms/token（32.7x）；gu 0.266 → 0.009（31.1x）
```
