# Qwen3.8-Flash-Next 第 46 轮实测记录

> 主题：**分块 prefill 第一步（T 个 token 一次算）的实现与验证**。
> 结果分两半：**分块算子层全部逐位等价**（DeltaNet / QSA / 单层前向），但**模型级驱动还有 bug**，且**当前形态比逐 token 更慢**——原因是基准跑在图里、分块路径全 eager。生产未启用（`FNX_CHUNK=1`）。

---

## 1. 已实现

| 文件 | 改动 |
|---|---|
| `fnx/engine.py` | 新增 `FnLayer._deltanet_chunk` / `_qsa_chunk` / `forward_chunk`；`_CHUNK = FNX_CHUNK`（默认 1 = 关闭） |
| `fnx/model.py` | 新增 `FnModel.prefill_chunk(ids)`（只推进状态，不出 logits） + `emb_chunk` 缓冲 |
| `fnx/server.py` | prefill 循环在 `FNX_CHUNK>1` 时按块调用；最后 token 仍走正常 `step()` |

分块做了哪些批量化：
- 投影全部 `M=T`（qkv/z/b/a 一次算子调用）——这是收益来源（第 45 轮实测摊薄 32.7×）
- 卷积改成"拼 3 列历史 + 4 个位移求和"（等价于逐 token 滑窗）
- KV 写从 one-hot 广播（O(P)）改成 `index_copy_` 一次写 T 行
- 注意力改成 `[T, NAH, P]` 的块矩阵 + 带状 mask
- DeltaNet 递推与 MoE 仍逐 token（顺序依赖 / 专家集合不同）

## 2. 验证：算子层逐位等价 ✅

| 对比项 | 结果 |
|---|---|
| `_deltanet` vs `_deltanet_chunk`（T=1，同一 h） | out / S / conv **全 0.000e+00** |
| `_qsa` vs `_qsa_chunk`（T=1，pos=5） | out / kv.k / kv.v **全 0.000e+00** |
| `L.forward` vs `L.forward_chunk`（T=1，L0 线性层） | out / S **全 0.000e+00** |
| `L.forward` vs `L.forward_chunk`（T=1，L3 全注意力层） | out / kv.k **全 0.000e+00** |

## 3. 还没过的两关

### 3.1 模型级驱动仍有偏差（未定位）

`tests/t_chunk_equiv.py`（60 token prompt，`FNX_NO_CAPTURE=1` 两边都 eager）：

```
logits maxdiff=1.264e+00  rel=6.903e-02   argmax 不一致
```

但**单层是逐位一致的**，所以问题在 `prefill_chunk` 这个驱动里（pos 记账 / ctx / 跨卡中转 / 缓冲复用之一）。
下一步的定位方法已经很清楚：比 `prefill_chunk([x])` 与 `step_pad(x, with_head=False)`（单 token、全模型），再逐层 dump 第一处偏差。

### 3.2 当前形态反而更慢（0.67×）

```
逐 token : 3657.6 ms → 16.4 tok/s   ← 走 NPUGraph（每 token 一次 replay）
分块 T=8: 5445.0 ms → 11.0 tok/s   ← 全 eager
```

原因不是分块没摊薄，而是**基准是图、分块是 eager**：45 层 eager 一个 chunk 要发几千次 launch，比图 replay 还贵。
⇒ **分块 prefill 必须配"分块图"**（按 T 再捕一张图，与 decode 的 g0 并存），否则收益全被发射吃掉。这是下一步的主要工作量。

## 4. 生产状态

**未启用分块**（`FNX_CHUNK` 未设置 → 默认 1）。服务已恢复并实测：

| 项 | 值 |
|---|---|
| TPOT | **63.7 / 64.2 ms** |
| mean_nll / ppl / top1 | **2.1648 / 8.713 / 48.0%** |
| dev0 显存 | 58.97 GiB |
| systemd / watchdog | active / 运行中 |

## 5. 下一步（按顺序）

1. **定位驱动 bug**：`prefill_chunk([x])` vs `step_pad(x, with_head=False)` 单 token 对比，逐层找第一处偏差。
2. **给分块路径加图捕获**：`capture()` 里多捕一张 `gc(T)`；`prefill_chunk` 改成 replay。这一步做完才有速度收益（预期重新回到"摊薄 30×"的账上）。
3. 再测 T=8/16/32 的 prefill tok/s，与第 45 轮的推算（40–45 tok/s）对照。
4. 最后做 MoE 并集/分组（第 45 轮 §2 的两条写法之一）。

## 6. 复现

```bash
# 算子层等价性（全部应为 0.000e+00）
docker exec fnx bash -lc 'cd /root/ascend-dev/fnx && FNX_NO_CAPTURE=1 ... \
  /root/ascend-dev/venv/bin/python tests/t_chunk_dn.py'      # DeltaNet
docker exec fnx bash -lc '... python tests/t_qsa_chunk.py'   # QSA
docker exec fnx bash -lc '... python tests/t_chunk_layer.py' # 单层

# 模型级（当前 FAIL）
docker exec fnx bash -lc 'FNX_NO_CAPTURE=1 FNX_CHUNK=8 ... python tests/t_chunk_equiv.py'
```
