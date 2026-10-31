# 第 48 轮实测记录：分块 prefill 实现 + 逐位精确性追查 + MoE 是唯一没摊薄的部分

日期：2026-10-31
目标：把 prefill 从 61–71 ms/token（14–16 tok/s）拉起来。
手段：按第 47 轮（§4.20）修正后的计划 —— ① 块 = 128（等长分块）② DeltaNet 递推分块 ③ 分块图捕获 ④ MoE 并集 + 分组 GEMM。
本轮完成：**第 ① 步（含块内 KV 批量写、banded causal mask）**，并把数值一致性追到**逐位相等**；实测出真瓶颈，修正了第 47 轮的收益预测。

---

## 1. 实现了什么

`FnLayer.forward_chunk(h, pos0, T)` / `_deltanet_chunk` / `_qsa_chunk` + `FnModel.prefill_chunk(ids)`：

| 部分 | 逐 token（生产） | 分块（本轮） |
|---|---|---|
| 稠密投影（16 个 GEMM/层） | M=1 每次读一遍权重 | **M=T 一次读一遍** |
| DeltaNet conv | 滑窗 cat + `.sum(-1)` | 3 列历史拼左边 + 4 位移求和 |
| DeltaNet 递推 | 每 token 一次 | **仍逐 token**（第 ② 步再做） |
| 全注意力 KV 写入 | one-hot 广播 | `index_copy_(1, pos, ...)` 一次写 T 行 |
| 全注意力计算 | 到 pos+1 为止 | `[T, pos0+T]` banded causal mask（切片只算必要列） |
| MoE | 每 token 10 专家 | **仍逐 token**（第 ④ 步再做） |
| 跨卡中转（dev1 3 层） | 每 token 一次 H2D | **每块一次** |

服务侧：`FNX_CHUNK>1` 时 `server.py` 用 `prefill_chunk` 走整块，**最后一个 prompt token 仍走 `step_pad` 出 logits**（不改变采样语义）。

---

## 2. 数值一致性：从 logits 偏 6.9e-2 追到 0.000e+00

第一版分块跑出来的 logits 与逐 token 差 `rel 6.9e-2`，argmax 都变了。逐层下钻（eager，`FNX_NO_CAPTURE=1`）：

1. 逐层 bisect：**45 层 dev0 + 3 层 dev1 全部 `rel 0.000e+00`** —— 但那是在 `pos=0`（S 全零）下做的，等于只验了"第一步"。
2. 全层状态指纹（S/conv/KV 的 sum）逐 token 对比：tok0、tok1 **0/84 差异**，tok2 突然 71/84 差异。
3. 给 `FnLayer.forward` / `forward_chunk` 打 hook 比 in/out：tok2 的 **L00 输入逐位相同（d=0.000e+00）、输出差 8.1e-3** ⇒ 偏差在层内部，不在 pos/缓存/中转。
4. 隔离到算子：非零状态下 `_deltanet` vs `_deltanet_chunk`，**y max 7.812e-03**（正好 1 个 bf16 ulp），`conv` 精确、`S` 差 1.2e-2。

**根因**：卷积 4 个 tap 的归约顺序不同。
- 逐 token：`(buf * conv_w).sum(-1)` —— torch 用**成对归约**；
- 分块第一版：`w0*x0 + w1*x1 + w2*x2 + w3*x3` —— **左结合**。

bf16 下两者差 1 ulp，再经 MoE 的 `topk` 边界（专家集合一换就跳变）放大成整句 logits 偏 6.9e-2。

**修法**：分块也用 `torch.stack([...], -1).sum(-1)`。修完：

```
=== ① 非零状态 T=1 算子等价 ===
  dn.y  max=0.000e+00   dn.S max=0.000e+00   dn.conv max=0.000e+00   qsa.y max=0.000e+00
=== ② 端到端（8 token，分块=1/4/8）===
  分块=1: L00.S max=0.000e+00    分块=4: 0.000e+00    分块=8: 0.000e+00
```

⇒ **分块 prefill 与生产逐 token 路径逐位等价**，不需要重新标定 NLL/MCQ 的口径。

> 方法论留档：**"逐层精确"这种结论必须说明状态是否为零**。零状态下 S=0 会让整条递推退化成一次外积，任何归约顺序差异都被掩盖。

---

## 3. 速度实测（256 token，eager 对 eager，真实权重）

| chunk | 全量 ms/token | tok/s | MoE 置零 ms/token | MoE 部分 |
|---:|---:|---:|---:|---:|
| 1（逐 token step） | 213.60 | 4.68 | 155.96 | 57.64 |
| 8 | 93.16 | 10.73 | 33.02 | 60.14 |
| 32 | 78.37 | 12.76 | 18.11 | 60.26 |
| 64 | 76.78 | 13.02 | ~17 | ~60 |
| **128** | **75.83** | **13.19** | **~16** | **~60** |

两条读法：

1. **稠密+注意力+DeltaNet 递推确实摊薄了 8.6×**（155.96 → 18.11 ms/token，T=32 起饱和）。
2. **MoE 完全不摊薄，恒定 ~60 ms/token** —— 因为 `forward_chunk` 里 MoE 仍是 `torch.cat([self._moe(mixed[t:t+1]) for t in range(T)])`，每 token 一次调用、每次读自己那 10 个专家的权重。它一项就吃掉 75.83 里的 79%。

**所以第 ① 步单独上线是负收益**：75.83 ms/token 比生产（图捕获的逐 token）61.7 ms/token **慢 1.23×**。生产 prefill 的 61.7 ms/token 里 MoE 同样占 ~60，稠密只占 ~1.7 —— 逐 token 路径的稠密部分早就不贵了（图捕获把发射开销吃掉了），分块省的是"本来就不贵"的那一段。

⚠️ 这推翻了 §4.19/§4.20 里"稠密 GEMM 12.7 ms → 0.4"的推算：**12.7 ms 是 r45 微基准里 eager 的每 token 成本，不是图捕获生产路径的成本**。生产路径稠密只有 ~1.7 ms/token。

---

## 4. 结论与下一轮

**结论：prefill 的唯一瓶颈是 MoE（~60 ms/token，decode 也一样）。** 稠密分块、DeltaNet 递推分块、等长分块都只能动剩下的 1.7 ms。这与 decode 的 TPOT 63.7–65.1 ms 完全自洽：prefill ≈ decode ≈ MoE 60 + 其余 4。

按第 47 轮的并集表（T=128 → 470 专家 / 9.0 MB per token，2.7×），MoE 分组做到位后：
- MoE 60 → ~5 ms/token（2.7× 权重 + 每层调用数从 T×10 降到 |U|）；
- 加上分块图捕获（去掉 eager 的宿侧空隙）：预期 **7–12 ms/token（60–100 tok/s）**。

下一轮按此顺序做：**④ MoE 并集 + 按专家分组的 GEMM（收益最大，且与 chunk 大小解耦）→ ③ 分块图捕获 → ② DeltaNet 32-token 递推**。

> 注意 ④ 的实现前提：把 `ExpW4` 的容器从 `int4pack [K,E,N/8]` 改成 `[E,K,N/8]`，让"单专家切片"是连续内存（否则每个专家都要 `contiguous()` 拷 4.1 MB，反而更慢）。

---

## 5. 生产验收（本轮结束时实测，含一个未解决的异常）

| 指标 | r46 记录值 | r48 本轮实测 | 判定 |
|---|---|---|---|
| `mean_nll`（eval_text 200 token） | 2.1648 | **2.1654** | ✅ 在 ±0.03 容差内 |
| `ppl` / `top1` | 8.71 / 48.0% | 8.718 / 48.5% | ✅ |
| 24 题单选题 | 24/24 | **24/24**（4.04 s/题） | ✅ |
| 稳态 TPOT（128 token 输出） | 63.7 / 64.2 ms | **73.6–74.1 ms（13.6 tok/s）** | ⚠️ **慢 15%** |
| dev0 显存 | 58.97 GiB | 59.53 GiB | ⚠️ 多 0.56 GiB |
| TTFT（前缀命中 20/21） | 0.08 s | 0.09 s | ✅ |

**关于 TPOT 异常**：本轮的代码改动只落在 `_deltanet_chunk` / `_qsa_chunk` / `prefill_chunk` 三个函数里，
`FNX_CHUNK=1`（生产配置）时**一行都不会执行**，所以理论上不可能影响 decode。已排除：
- 不是热降频：停机 5 min 后复测仍 75.1–75.3 ms；
- 不是脏状态：`systemctl stop` + `start` 干净重启后仍是 73.6–74.1 ms（4 次连续测量方差 <1 ms）；
- 不是宿侧空隙：`/fnx/prof` 的 `tok_decode_ms` 只有 9.85 ms/128 token（可忽略）；
- 不是外部负载：宿主 load average 1.3，64 核，服务进程仅占 0.53 核。

`hbm0=59.53 GiB` 也在**上一版代码**（看护自动重启的那次）的日志里出现，说明显存差异早于本轮改动。
⇒ 这是一个**独立于本轮工作的既有偏移**，需要在下一轮用"r47 代码 + msProf 设备/宿侧拆分"做 A/B 才能定位。
**在它查清之前，本仓库不再宣称 63.7 ms 这个 TPOT。**

## 6. 复现命令

```bash
# 分块等价性（非零状态 + 端到端）
bash /root/ascend-dev/p337_state.sh
# 吞吐扫描（含 MoE 置零对照）
bash /root/ascend-dev/p338_speed.sh
# 生产验收（NLL + TPOT）
bash /root/ascend-dev/p339_prod.sh
# TPOT 单测（4 次）
bash /root/ascend-dev/p340_tpot.sh
# MCQ
docker exec fnx bash -lc "cd /root/ascend-dev/fnx && /root/ascend-dev/venv/bin/python tools/mcq_eval.py"
```

本轮改动文件：`backend/fnx/fnx/engine.py`（`forward_chunk`/`_deltanet_chunk`/`_qsa_chunk` + conv 归约修正 + 注意力切片）、`backend/fnx/fnx/model.py`（`prefill_chunk` + `emb_chunk` 按需扩容）、`backend/fnx/fnx/server.py`（`FNX_CHUNK` 分支）。
探针：`tests/t_chunk_driver.py`、`t_chunk_fp.py`、`t_chunk_hook.py`、`t_chunk_state.py`、`t_chunk_speed.py`。

