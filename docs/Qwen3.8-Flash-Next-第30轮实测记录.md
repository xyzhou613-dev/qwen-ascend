# Qwen3.8-Flash-Next 第 30 轮实测记录

> 本轮的关键词：**把"每请求 5.7 s 固定开销"这个谜团彻底拆开——它根本不是开销，是 45 个可缓存的模板 token。**
> 顺带挖出模板里的 `reasoning_effort` / `enable_thinking` 开关：`enable_thinking=False` 让同样三个问题从 50.7 s 降到 22.6 s。
> 收尾时又补回被漏掉的**最后 2 层**：`mean_nll` 3.1874 → **2.1756**，掉字母现象消失；
> 修掉**停止符用成 pad** 的 bug（回答不再续写下一轮模板），再把 `lm_head` 搬到 dev1 换出 1 层捕获：
> **TPOT 166.3 → 142.9 ms**。
>
> 全部数字来自 <ascend-host> 真实服务进程（`/fnx/prof` 分段计时），不是估算。

---

## 0. 一句话结论

**没有任何"框架固定开销"。** 第 29 轮末尾观察到的 `TTFT ≈ 102 ms × prompt_len + 5.7 s`，
那个 5.7 s = **聊天模板静态前缀 45 token × 127.5 ms = 5.74 s**，误差 40 ms。

三件事由此而来：

| 改动 | 效果（实测） |
|---|---|
| 前缀状态缓存（45 token 只跑一次） | 「你好」8 token：**7.79 s → 2.03 s**（流式），逐位等价 |
| 对话级前缀缓存（2 槽，主机内存） | 3 轮对话第 3 轮：**10.23 s → 5.24 s**，答案逐字相同 |
| `enable_thinking=False` | 同 3 个问题：**50.71 s → 22.58 s**，输出还更干净 |
| **★补回最后 2 层（46 → 48 层）** | `mean_nll` **3.1874 → 2.1756**（ppl 24.23→8.81），掉字母现象消失 |
| **★停止符修复（`<|im_end|>`）** | 回答不再续写 `\nuser\nassistant\n<think>`，顺带每请求快约 2 s |
| **★head 搬 dev1 + 46 层捕获** | TPOT **166.3 → 142.9 ms**（−14%），质量不变 |

质量口径：同一 `/fnx/loss`（200 token）下 **`mean_nll = 2.1756`**、`top1 = 49.00%`、`median_rank = 1`。
前缀/对话缓存与模板改动本身**不改质量**（等价性见 §3.4/§4）；第 4 行是**层数**带来的真实提升；
head 搬运与层数分配同样不改质量（§13.1 表里 `mean_nll` 三个配置完全一致）。
设备显存：`hbm0 59.54 GiB`（46 层捕获）、`hbm1 3.19 GiB`（2 层 eager + lm_head/mixer）。

---

## 1. 定位：5.7 s 到底是什么

### 1.1 分段计时（`FNX_PROF=1`，插入 `torch.npu.synchronize()` 才是设备真值）

`tests/t_profile_req.py` + 新增 `GET /fnx/prof`。两个 prompt 各一次：

| 段 | 「你好」53 token / 8 new | 长 prompt 85 token / 4 new |
|---|---|---|
| `reset()`（清零 S/conv/KV + sync） | **1.33 ms** | 1.27 ms |
| prefill（逐 token） | **6630.12 ms** = 52 步 × 127.5 ms | **11188.26 ms** = 84 步 × 134.8 ms |
| decode / token | 134.17 ms（首步 137.66） | 142.68 ms |
| 采样 `argmax().item()`（D2H 同步） | **0.71 ms** | 0.22 ms |
| `apply_chat_template` | 0.37 ms | 0.34 ms |
| `tok(text)` 分词 | 1.81 ms | 0.95 ms |
| HTTP 墙钟（curl） | 7790 ms | 11777 ms |

**结论：prefill 每 token 和 decode 一样贵（127–135 ms），所以"固定开销"其实全是逐 token 的 prefill。**
`reset()` 1.3 ms、采样 0.7 ms、分词 2 ms、HTTP ~20 ms —— 这些以前被怀疑的东西全都可以排除。

### 1.2 那「你好」为什么是 53 个 token

```
[  0] 248045 '<|im_start|>'
[  1]   8678 'system'
[  2]    198 '\n'
[  3]  24342 'Reason'   ... 'effort is set to xhigh. Please think carefully
                            through the task, validate key assumptions,
                            consider plausible alternatives, and prioritize
                            correctness, consistency, and clarity in the final answer.'
[ 40] 248046 '<|im_end|>'
[ 41]    198 '\n'
[ 42] 248045 '<|im_start|>'
[ 43]    846 'user'
[ 44]    198 '\n'
[ 45] 109266 '你好'            ← 用户内容只有这一个 token
[ 46] 248046 '<|im_end|>'
[ 47]    198 '\n'
[ 48] 248045 '<|im_start|>'
[ 49]  74455 'assistant'
[ 50]    198 '\n'
[ 51] 248068 '<think>'
[ 52]    198 '\n'
```

**45 个 token 是模板 boilerplate（其中 38 个是那段 system 提示）**，两个不同用户内容的公共前缀
实测正好 = **45 token**（`tools/tok_prefix.py`）。45 × 127.5 ms = **5.74 s** —— 与观测到的 5.7 s 对账。

---

## 2. 模板里还有开关：`reasoning_effort` / `enable_thinking`

模板（`tools/tok_effort.py` 扫描）支持：

| 变体 | 静态前缀 | 说明 |
|---|---|---|
| 默认（`xhigh`） | **45** token | 第 50 行 `resolved_reasoning_effort == 'xhigh'` 才注入那段长 system 提示 |
| `reasoning_effort='low'` | 33 token | 换成一句较短的提示 |
| `reasoning_effort='medium'` | **3** token | `reasoning_instructions` 为空 → system 轮整个不渲染 |
| `enable_thinking=False` | **3** token | 同上，且 assistant 轮直接给结论，不产生 `<think>` 轨迹 |

---

## 3. 实现：前缀状态缓存

### 3.1 为什么可行

`reset()` 清零的东西是**全部**递归状态：每层 `S`（ΔNet）、`conv`、`ple_conv`、两卡的 `kv0.k/v`、`kv1.k/v`、
`pos0/pos1`、`ctx`。所以只要把这套东西快照下来、用时原样写回，就等价于"重新跑了那 N 个 token"。
（`engine.KVStore` 只有 `k/v` 两个列表，无隐藏长度计数器，已核对。）

### 3.2 新增接口（`fnx/model.py`）

| 方法 | 作用 |
|---|---|
| `save_state()` | 克隆全部递归状态（151.9 MiB） |
| `load_state(st)` | 直接覆盖（**不先 reset**），实测 2.5–4.2 ms |
| `save_into(st)` | 就地写入已分配槽位，避免每请求 alloc |
| `state_to_host(st)` | 快照搬到主机内存（单卡只剩 ~1.9 GiB，152 MiB × 3 太奢侈） |
| `state_bytes(st)` | 统计大小 |

### 3.3 切前缀的办法（`server.py::_build_prefix`）

用哨兵串渲染一次模板，取哨兵之前那段文本 → 单独分词 → 就是静态前缀：

```python
text = tok.apply_chat_template([{"role":"user","content":"\u0002FNXPFX\u0002"}], ...)
pids = tok(text[:text.find(SENT)]).input_ids     # 45 token
```

**安全闸门**：每次请求先做 `prompt_ids[:45] == prefix_ids` 的逐位比较，不相等就退回 `reset()` + 全量 prefill。
宁可慢，不可错。

### 3.4 等价性证据（这是上线前提）

`tests/t_prefix_cache.py`（A = reset + 全量 prefill，B = load_state + 剩余 prefill，比较首个 logits）：

| prompt | n | 剩余 prefill | maxdiff | argmax | 耗时 |
|---|---|---|---|---|---|
| 你好 | 53 | 7 | **0.000e+00** | 97237 / 97237 | 6693 → **1018 ms** |
| 请详细解释一下深度学习里的… | 64 | 18 | **0.000e+00** | 109849 / 109849 | 8097 → **2406 ms** |
| 1+1等于几？ | 58 | 12 | **0.000e+00** | 97237 / 97237 | 7329 → **1651 ms** |

逐位相同（不是"接近"）。连发 5 次：**1944 ms/次** vs 全量 **7612 ms/次**（3.9×）。

---

## 4. 对话级缓存：key 必须取 prefill 结束点

第一版把"生成结束时的状态"按 `prompt + 生成 token` 存，**一次都没命中**（`prefix_hit` 一直 45）。
原因（`tools/tok_prefix.py` 实测公共前缀 = 52，而不是 53）：

- 生成用的 prompt 尾巴是 `…assistant\n<think>\n`（`<think>` + 换行）
- 下一轮模板把历史渲染成 `…assistant\n<think>` + 回答内容（**没有那个换行**）
- → 两条 token 流在**第 52 位**就分叉，结束状态永远接不上

正确做法：**在 prefill 结束点存快照，key = `prompt_ids[:-1]`**。因为 prefill 只喂 `prompt[:-1]`
（最后一个 prompt token 由 decode 第一步喂），这一段的末尾恰好是 `…assistant\n<think>`，
逐位等于下一轮 prompt 的历史部分。

槽位管理：2 槽、预分配、`save_into` 就地覆盖、淘汰最短的键。**快照全部在主机内存**，
所以设备显存回到 `57.56 GiB`（与第 29 轮完全一致），每请求多花约 10 ms 的 H2D。

实测（3 轮对话，思考关之前）：

| 轮 | prompt | 改前 | 改后 | 命中 |
|---|---|---|---|---|
| 1 | 53 | 2.53 s | 2.51 s | 45 |
| 2 | 85 | 6.64 s | 5.68 s | 52（省 7 token） |
| 3 | 113 | **10.23 s** | **5.24 s** | 84（省 32 token） |

**答案逐字一致**（`/tmp/conv_2.txt` vs `/tmp/conv_0.txt`，三轮全 `相同`）——缓存没有改变任何输出。

---

## 5. A/B：缓存对 decode 零代价

`FNX_PREFIX_CACHE` / `FNX_CONV_CACHE` 全关 vs 全开，同一 prompt（出 92 token），各 2 次：

| 配置 | TTFT | 总 | TPOT |
|---|---|---|---|
| 缓存关 | 6.89 s / 6.76 s | 19.38 / 19.21 s | **137.3 / 136.8 ms** |
| 缓存开（首跑） | 1.21 s | 13.71 s | **137.3 ms** |
| 缓存开（命中） | **0.15 s** | 12.64 s | **137.3 ms** |

→ 缓存只影响 prefill，decode 一分不差。

### ★ 修正第 29 轮的一个数

第 29 轮报的 `TPOT 125.29 ms` 是三点法（8 token 8.195 s、32 token 11.202 s）解出来的，
现在有了**逐 token 直接计时**：`134.17 / 135.82 / 136.5 / 137.3 / 142.68 ms`，
200-token 长输出稳态 **133.1–145.2 ms（典型 136 ms）**。**真实 TPOT ≈ 136 ms，不是 125 ms**。
（三点法本身没算错，它把 prefill 估成 7.19 s、TPOT 125 ms；这组解在直接测量面前不成立。）

---

## 6. 四个模板变体实测（各 3 个问题，非流式，放开发挥）

| 变体 | 前缀 | 总时长 | 输出 token | 特点 |
|---|---|---|---|---|
| `xhigh`（默认） | 45 | 50.71 s | 319 | 思考冗长，`</think>` 混在正文里 |
| `low` | 33 | 55.96 s | 323 | 思考依旧长 |
| `medium` | **3** | 50.56 s | 315 | 前缀没了，思考还是长 |
| **`enable_thinking=False`** | **3** | **22.58 s** | **112** | 直接给结论，**还最干净** |

`no-think` 的样本（同一模型、同一权重、只改模板参数）：

```
Q: 3加5等于几？只回答数字
A: 8

Q: 你好，请用一句话介绍你自己
A: 我是通义千问（Q），由通义实验室独立开发，旨在成为你思考时最得力的伙伴。

Q: 用三句话解释什么是 MoE（混合专家）模型
A: Mo（混合专家）模型是一种由多个"专家"子网络和一个"gating"（门控）机制组成的稀疏架构。
   门控机制负责根据输入动态选择少数几个最相关的专家进行计算，而非激活所有专家。
   这种设计在保持模型总参数量巨大的同时，仅激活部分参数，从而在提升模型容量和性能的同时显著降低计算成本。
```

对比 `xhigh` 下的 MoE 回答（同一问题）：`'我们需要回答用户：用三句话解释什么是 Mo（混合专家）模型。需要三句话， likely Chinese. already doing. Need final only three sentences…'`
—— 思考轨迹把 `max_tokens` 全吃掉了，正文根本没出来。**所以思考开着既慢又差。**

---

## 7. 生产定稿

```bash
# /root/ascend-dev/p275.sh（= backend/remote/p275_prod.sh）
FNX_PACK=0 FNX_EXP_G=64 FNX_MOE_KERNEL=0 FNX_NO_PLE=1 FNX_PACKED_TORCH=1 \
FNX_LPC=46 FNX_NLAYERS=48 FNX_PORT=8000 \
FNX_PREFIX_CACHE=1 FNX_CONV_CACHE=2 FNX_PROF=0 FNX_ENABLE_THINKING=0 \
/root/ascend-dev/venv/bin/python -m fnx.server
```

冷启动 **30 s**（装配 10.0 s + 图捕获 2.3 s + 前缀缓存 0.5 s）。
布局：dev0 = **46 层图捕获**（59.54 GiB）；dev1 = **2 层 eager + lm_head/mixer**（3.19 GiB）。

生产实测（48 层 / 46 捕获 / 思考关）：

| 场景 | 耗时 | 备注 |
|---|---|---|
| 「你好」（prompt 13 token） | **3.22–3.45 s** | 9 步 prefill + 约 13 个输出 token |
| 「你好」（同 prompt 再来） | **0.31 s** | 前缀+对话缓存全命中，prefill=0 |
| 20 请求混合压测 | 76.7 s（20/20 成功） | 显存持平、错误 0 |
| 3 轮对话 | 3.24 / 6.55 / 3.99 s | `prefix_hit` 3 / 3 / 44 |
| 120 token 输出 | — | **TPOT 142.9 ms** |

`/fnx/prof` 现在会给出 `prefix_hit` / `n_prefill` / 变体表（`FNX_PROF=0` 时只给计数，不给会误导的"入队时间"）。

**为什么 `FNX_PROF=0`**：开 `FNX_PROF=1` 会在每 token 前后加 `torch.npu.synchronize()`（为了拿设备真值），
排查完就关；计数（`prefix_hit`/`n_prefill`）在 PROF 关时依然保留。

---

## 8. 已知问题（本轮只记录，不修）

1. **`MoE`→`Mo`、`Qwen`→`Q` 掉字：根因是"只有 46 层"，补到 48 层后消失。**
   排查链条（全部实测）：
   - 分词器洗清：`MoE` = `[24797, 36]`、`Qwen` = `[48, 16451]`，往返解码完全正确（`tools/tok_moe.py`）；
   - head 洗清：把 int8 lm_head 换成原始 bf16（1.18 GiB）后 `mean_nll` 3.1874 → 3.1865（**没有差别**），
     探针 rank 一模一样（`tests/t_head_ab.py`）；
   - 定点探针显示模型在 `Q` 之后对 `wen` 是 **rank 1、argmax 是 `）`** —— 它自己在两种续写间摇摆，
     典型的"层数不够导致表示不足"，不是解码 bug；
   - 补回最后 2 层（48 层）后同一句话输出 `我是通义千问，由阿里巴巴通义实验室研发…`，
     `MoE（混合专家模型）… 而 Qwen（通义千问）是…` —— **两个词都完整了**。
2. **长 prompt prefill 仍是逐 token 串行**（约 130–170 ms/token）：76 token 的 prompt 要 ~10 s。
   要突破必须做 chunked prefill，而 ΔNet 是**对位置递归**的，需要分块矩阵形式 —— 属于研究级工作。
3. **TPOT 142.9 ms 是每 token 前向本身**的墙：46 层图 + 2 层 eager + 跨卡中转，MoE 自定义 kernel 还没启用。
   剩下的 2 层 eager 各约 12 ms（`LPC` 从 44→45→46 的实测差：166.3→154.4→142.9）。
   `LPC=47` 只差 256 MiB 就 OOM（dev0 到 60.04/60.96 GiB），要再省得动权重布局。
4. 第 29 轮遗留：PLE 为何有害仍未定性；W4 kernel 深层的精度差（同放置 `mean_nll` 11.82 vs torch 3.19）未定位。

---

## 9. 下一轮建议（按性价比排序）

1. **继续往质量上压**：48 层已经把 `mean_nll` 打到 2.1756。现在 sample 一下还能不能再省：
   `FNX_EXP_G` 64→32 需要重跑 `repack_v2`（48 层 860 s，盘还有 2.6 T），口径就用 `/fnx/loss` ——
   第 28 轮 §18 说"量化误差主因是 g64 分组（4.81%）"，值得在**生产路径**上复验一次。
2. **把最后 4 层 eager 的钱省掉**：现在 166 ms 里约 70–80 ms 是 dev1 上那 4 层 eager。
   可行方向：dev0 腾出 ~2 GiB（前缀/对话快照已在主机；或降 O(1) 图池）多捕 1–2 层，
   或试 48 层的专家 offload。**每搬回 1 层约省 18 ms（10%）。**
3. **长 prompt 的 chunked prefill**（若要做长文档场景）：ΔNet 分块矩阵形式是研究级工作；
   短期绕法是"多请求 B>1 batch"而不是单请求长 prompt。
4. **MoE kernel**（`FNX_MOE_KERNEL=1` 目前关着）：先在 8 层配置对拍 + 计时再决定。
   注意第 29 轮 §10：同布点下 kernel 路径 `mean_nll` 差 8.6 nats，深层精度未定位。
5. **head bf16 不要再试**：本轮已证否（+1.18 GiB 显存换 −0.0009 nats，top1 还降了 0.5 个点）。
6. **明确不要做的**：dev1 图捕获（已用平凡链证否）；稠密权重压 4bit（深层层量化风险）；
   `torch.compile`/torchair（未验证，风险高）。

---

## 10. ★补回最后 2 层：一条被忽略的 1 nats

### 11.1 先排除 head：bf16 lm_head 没有任何用

`lm_head` 现在是 int8（`q8/s8`，`E.QW` 分块反量化）。怀疑它把小众 token 压掉了，于是把原始 bf16 权重
（`model-00131-of-00131.safetensors`，`shape=(248320, 2560)` bf16，1.18 GiB）换上去对照
（`tests/t_head_ab.py`，head 不入图，运行中可换）：

| 口径 | int8 head | bf16 head |
|---|---|---|
| `mean_nll` | 3.1874 | **3.1865（−0.0009）** |
| `top1` | 48.50% | 48.00%（**反降**） |
| 探针 `E`(36) | NLL 1.036 / rank 0 | NLL 1.035 / rank 0 |
| 探针 `Q`(48) | NLL 1.065 / rank 0 | NLL 1.066 / rank 0 |
| 探针 `wen`(16451) | NLL 3.817 / **rank 1，argmax `）`** | NLL 3.818 / rank 1，argmax `）` |

**结论：head 精度不是瓶颈**（多花 1.18 GiB 显存换 −0.0009 nats，不划算，已放弃）。
但探针暴露了真问题：模型在 `Q` 之后对 `wen` 的排名是 **1**、argmax 是 `）`
—— 它自己在"拼完 Qwen"和"立刻收括号"之间摇摆，这是**表示不足**，不是解码/量化格式问题。

### 11.2 真正的原因：只有 46/48 层

第 29 轮的参考深度曲线（`...第29轮实测记录.md` §2）显示信号几乎全在最后几层：
0–38 层 ≈12.4 平，39→10.96，42→9.80，45→7.91，**48→5.4437**。

我们之前一直是 `FNX_NLAYERS=46`（44 捕获 + 2 eager）——**最后 2 层根本没跑**。
当时 §0.4 把 "48/48 = 44+4" 列为"可选、性价比一般（+26 ms）"，但**从没量过它的质量收益**。

实测（`p283_layers48.sh`，同一 `/fnx/loss` 口径、同一 eval_text 前 200 token）：

| 配置 | mean_nll | ppl | top1 | median_rank | TPOT | hbm1 |
|---|---|---|---|---|---|---|
| 46 层（44 捕获 + 2 eager） | 3.1874 | 24.23 | 48.50% | 1 | 133.1–145.2 ms | 2.59 GiB |
| **48 层（44 捕获 + 4 eager）** | **2.1756** | **8.81** | 49.00% | 1 | **166.3–176.5 ms** | 5.17 GiB |

**−1.01 nats / ppl −64%，代价只有 +33 ms/token（+25%）。这是本轮收益最大的一项。**

输出质量的实际变化（同一提示词、同一权重、只差这 2 层）：

| | 46 层 | 48 层 |
|---|---|---|
| 自我介绍 | `我是通义千问（Q），由通义实验室独立开发…` | `我是通义千问，由阿里巴巴通义实验室研发的超大规模语言模型…` |
| MoE/Qwen | `Mo（混合专家）模型…` | `MoE（混合专家模型）… 而 Qwen（通义千问）是阿里巴巴通义实验室自主研发的大语言模型系列。` |

掉字母现象**完全消失**。所以这条"小 bug"其实是深度不足的冰山一角 ——
**"装不下 48 层"是误读：装不下的是"48 层全部图捕获"，48 层里 4 层走 dev1 eager 完全装得下。**

### 11.3 48 层下缓存同样成立

线上 3 轮对话（思考关，prompt 13 / 45 / 73）：

| 轮 | prompt | prefix_hit | n_prefill | 墙钟 |
|---|---|---|---|---|
| 1 | 13 | 3 | 9 | 4.08 s |
| 2 | 45 | 12 | 32 | 7.84 s |
| 3 | 73 | 44 | 28 | 6.45 s |

前缀状态在 48 层是 159.0 MiB（46 层时 151.9 MiB），对话缓存照常命中（12 → 44）。

---

## 11. ★停止符 bug：`NG_EOS` 是 pad，不是 eos

修 48 层之后，输出里出现新的破绽（第 11 节修完之后才发现）：

```
Q: 你好
A: '你好！很高兴见到你。请问有什么我可以帮你的吗？\nuser\nassistant\n<think>'
```

**说完话之后继续"编造下一轮模板"**。查 `tokenizer` 特殊 token：

| id | 含义 |
|---|---|
| 248044 | `<|endoftext|>`（也是 `pad_token_id`） |
| 248045 | `<|im_start|>` |
| **248046** | **`<|im_end|>`（`eos_token_id`，对话轮真正的结束符）** |

而 `engine.NG_EOS = 248044` —— 是 **pad**。`generate()` 里只写了 `if nxt == E.NG_EOS: break`，
**从来没有在 `<|im_end|>` 上停过**，所以模型把轮尾当普通文本继续往下写。

修法（`server.py`）：停止符集合 = `{248044, 248046, tokenizer.eos_token_id}`，
外加尾部清理正则（只在**行首角色名 / 模板标记**处裁，避免误伤正文里的 assistant/system 单词）：

```python
_TAIL_RE = re.compile(
    r"(?:\s*(?:<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<think>|</think>)\s*"
    r"|\n\s*(?:user|assistant|system)\s*(?=\n|$))+$")
```

效果（同一批 20 个请求）：

| | 修前 | 修后 |
|---|---|---|
| 「你好」非流式 | 5.29 s，尾部带 `\nuser\nassistant\n<think>` | **3.22–3.45 s，干净** |
| 「1+1=?」 | `'1 + 1 = **2**\n'` | `'1 + 1 = **2**'` |
| 短诗 | `…月沉入海心。\n'` | `'风止于林梢，\n月沉入海心。'` |

顺带快了约 2 s —— 因为不再白跑那 8 个模板 token。

---

## 12. 把 eager 层的钱省回来：head 搬到 dev1 + 46 层捕获

### 12.1 扫描：dev0 还能多捕几层

`44 层是上限` 的旧结论只在"head 也在 dev0"时成立。先直接扫 `FNX_LPC`：

| LPC | dev0 | dev1 | dev0 alloc | mean_nll | TPOT |
|---|---|---|---|---|---|
| 44 | 44 捕获 | 4 eager | 57.56 GiB | 2.1756 | **166.3 ms** |
| 45 | 45 捕获 | 3 eager | 58.85 GiB | 2.1756 | **154.4 ms** |
| 46 | 46 捕获 | 2 eager | — | — | **OOM（只差 82 MiB）** |

`mean_nll` 三个配置完全一致 —— **层放哪不影响数学**，只影响速度。

### 12.2 把 int8 lm_head + mixer 搬到 dev1

`lm_head` 是 int8（`q8`+`s8`，约 646 MiB）常驻 dev0，而 dev1 那时只用了 3.88 GiB。
搬到 dev1 后：

- dev0 释放 646 MiB → 腾出第 46 层的位置；
- **附带收益：layers1 跑完后隐藏状态本来就在 dev1**，head 在 dev1 算可以省掉每 token
  **两次跨卡中转**（`h1 → host → h0`），实测额外省 ~3.5 ms。

实现：`FNX_HEAD_DEV`（默认 auto：有 dev1 层就放 dev1），`self.logits` 跟着 head 走，
`head()` 在 dev1 时跳过中转，`gh` 图捕获仅在 head 位于 dev0 时启用（dev1 捕获已被证否）。

重扫：

| LPC | dev0 / dev1 alloc | mean_nll | TPOT |
|---|---|---|---|
| 45 | 58.25 / 4.49 GiB | 2.1756 | 150.9 ms |
| **46** | **59.54 / 3.19 GiB** | **2.1756** | **142.9 ms** |
| 47 | — | — | OOM（需 802 MiB，仅剩 546 MiB） |

**定稿 LPC=46：TPOT 142.9 ms，比最初的 166.3 ms 快 14%，质量一模一样。**

### 12.3 压力验收（dev0 只剩 900 MiB 余量，必须确认不越跑越涨）

20 个混合请求（长短/流式/多轮，含 3 轮同一个对话）：

- 20/20 成功，墙钟 76.7 s，`server.log` 错误 0；
- **hbm0 全程 61727 MB 不动**（起始 = 结束），hbm1 3781 → 4027 MB（eager 层临时缓存，稳定）；
- 多轮：turn1/turn2/turn3 = 3.24 / 6.55 / 3.99 s，`prefix_hit` 3 / 3 / 44；
- `/fnx/loss` 仍是 **2.1756**。

---

## 13. 工程化：把"服务老是被杀"查到底（三个真 bug）
### 13.1 现象

连续压测（24 题 MCQ）时服务总是跑到一半就"没了"。日志里没有 traceback，只有 uvicorn 的
`Shutting down / Application shutdown complete` —— 说明是**收到 SIGTERM 优雅退出**，不是崩溃。

### 13.2 根因一（最要命）：`async def` 里做同步生成，把事件循环堵死

```python
@app.post("/v1/chat/completions")
async def chat(req: Request):
    ...
    with LOCK:
        toks = list(eng.generate(...))     # ← 几秒到几十秒的同步计算，跑在事件循环线程上
```

生成期间**事件循环完全阻塞** → 连 `GET /health` 都没人处理 → 看护脚本 20 s 探一次、连续 3 次超时
→ 判定"服务挂了" → `pkill` + 重启 → **把正在跑的请求连服务一起杀**。
24 题的压测正好在第 23 题被杀，与"看护 60 s 判定窗口"完全对上。

修法：生成放进线程（`await asyncio.to_thread(_work)`）。
**实测：29.1 s 的长生成期间，`curl /health` 立即返回 `{"status":"ok"}`。**

### 13.3 根因二：流式请求实际上没有持锁

```python
with LOCK:
    ...
    return StreamingResponse(sse(), ...)   # 生成器是之后才被消费的，此时 LOCK 已经释放
```

`eng.generate()` 是生成器，`sse()` 被 starlette 在**线程池里**迭代，而 `with LOCK` 在
`return` 时就已经退出了 → **两个流式请求会同时改同一套 ΔNet 递归状态 / KV**（结果会是乱的）。
修法：`LOCK.acquire()` 放进 `sse()`，`finally` 里释放，覆盖整个生成期。

### 13.4 根因三：重启链路自己会骗自己

| 坑 | 现象 | 修法 |
|---|---|---|
| `pgrep -f 'python -m fnx.server' \|\| true` | `\|\|` 让 bash 保留 shell，pgrep **匹配到自己** → 永远报"有残留" | 模式写成 `[p]ython` |
| `pgrep -f` 同时匹配 `bash -lc` 包装进程 | 一个服务显示 `count=2`，误判成双实例 | 判活看 `/health`，不看计数 |
| `rm -f server.log` | 崩溃现场被自己删掉 | 改 `mv` 轮转（保留 `server.log.prev`） |
| 看护在装配窗口（~30 s）内判定失败 | 与手工重启互相 pkill，服务被双方都杀掉 | `last_restart` 时间戳 + 150 s 宽限期 |
| 进程还活着但 `/health` 不通 | 看护立刻重启，杀掉正在跑的请求 | 进程存活时改为连续 15 次（5 min）才动手 |

### 13.5 新增运维面

- **`GET /metrics`**（Prometheus 文本）：`fnx_requests_total`、`fnx_tokens_out_total`、
  `fnx_prefill_tokens_total`、`fnx_prefix_hit_tokens_total`、`fnx_queue_ms_avg`、
  `fnx_ttft_ms_avg`、`fnx_tpot_ms_avg`、`fnx_prefill_hit_ratio`。
- **`/root/ascend-dev/watchdog.sh`**：20 s 探活、连续 3 次失败重启、`mkdir` 锁防并发重启、
  重启后 150 s 宽限期、进程存活时放宽到 15 次。
- **`/root/ascend-dev/p275.sh`（v5）**：先停看护 → 杀进程并**等它真退出** → 日志轮转 →
  写时间戳 → 起服务 → 等 `/health` → 拉看护。单一入口。
- **容器主进程化（尝试过，暂时回滚）**：把 server 提为容器 PID 1（沿用 fnx 的镜像/设备/挂载/环境）
  可以彻底摆脱"`docker exec -d` 孤儿进程"，但**设备不能被两个容器共享**
  （`The resources are busy`），而当时旧容器里还跑着参考任务 → 等它跑完再做。

### 13.6 并发正确性 & 缓存不变性

**并发**（2 个流式 + 1 个非流式同时打）：总墙钟 18.06 s，三个客户端各自墙钟 4.40 / 12.45 / 18.06 s
（含排队），输出全部连贯 —— **完全串行化**，没有出现"两个流互相改状态"的乱码。
（注：一开始我把判据写成"总墙钟 ≈ 客户端耗时之和"，那是错的：客户端耗时含排队时间，会重复计算。）

**缓存不变性**（排除缓存污染）：同一 prompt 在"缓存热"和"刚重启、缓存全清"下，
回答**逐字相同**：

```
热态 A1 = 热态 A2 = 冷态 B = "“张量”并非标准的数学或物理学术语，通常是指**张量（Tensor）**的误写…"
```

顺带说明：这个回答本身是**错的**（张量当然是标准术语），但它是模型自己在
`top1 = 49%` 水平上的判断失误，不是缓存/框架的问题 —— 需要靠量化精度或层数解决，不是靠修 bug。


---

## 14. 任务级评测：24 题单选题

`tools/mcq_eval.py`（固定题库、强制只答字母、`enable_thinking=False`）：

```
=== 准确率 24/24 = 100.0%   平均 7.48 s/题 ===
```

**别把这个 100% 当能力结论**——题库是常识/算术/计算机基础，难度刻意压低，它的用途是
**功能门槛**：换量化、换层数、换内核之后先过这一关，再谈 `/fnx/loss` 的 nats。

每题 7.48 s 也解释了延迟结构：每题 prompt 约 50 token、静态前缀只有 3 token（思考关），
prefill 就是 47 × ~143 ms ≈ 6.7 s —— **prefill 才是交互延迟的主项**，不是 decode。

（注：这批数字是在参考 bf16 任务占满 CPU 时测的，绝对延迟略有膨胀；准确率不受影响。）

---

## 15. ★我们离 bf16 到底多远（配对口径，第一次拿到）

前面的 `mean_nll = 2.1756` 一直是**孤立**的数（第 29 轮的参考 5.4437 是 T=3 口径，不可比）。
这次用参考实现（原始 bf16、CPU、层主序流式）在**同一段 eval_text 的同一 token 前缀**上算，
再让线上 `/fnx/loss` 算同一 12 个预测，做**逐 token 配对**。

代价：CPU 上 48 层 × 12 token = **4100 s（68 min）**（每层约 91 s，主要耗在 CPU 上的 MoE）。

| 位置 | 参考 bf16 NLL | rank | 线上 W4 NLL | rank | 差 |
|---|---|---|---|---|---|
| 0 | 1.377 | 0 | 1.788 | 0 | +0.411 |
| 1 | 5.541 | 28 | 5.905 | 37 | +0.365 |
| 2 | **9.414** | **891** | 9.481 | 958 | +0.068 |
| 3 | 5.725 | 40 | 5.539 | 39 | −0.186 |
| 4 | 6.457 | 40 | 7.898 | 62 | **+1.441** |
| 5 | 2.032 | 1 | 2.038 | 1 | +0.006 |
| 6 | 1.013 | 0 | 1.168 | 0 | +0.155 |
| 7 | 0.038 | 0 | 0.076 | 0 | +0.038 |
| 8 | 3.665 | 1 | 3.696 | 3 | +0.031 |
| 9 | 0.524 | 0 | 0.377 | 0 | −0.146 |
| 10 | 3.705 | 8 | 4.746 | 15 | **+1.041** |
| 11 | 4.804 | 18 | 4.757 | 25 | −0.046 |
| **均值** | **3.6911** | | **3.9558** | | **+0.2648** |

**结论要分两句话看：**

1. **中位数只有 +0.0528 nats** —— 大多数 token 上我们的 W4 和 bf16 几乎没差；
   rank 完全相同的 5/12。所以**量化不是主要矛盾**。
2. **均值 +0.2648 是被两个离群点拉起来的**（pos 4 +1.44、pos 10 +1.04）。
   W4 在少数 token 上会明显退化，这类 token 正是"模型答错"的高发位置。

**还有一个更重要的发现**：连参考 bf16 自己在这段文本上都只有 `mean_nll = 3.6911`、
`top1 = 33.33%`、`median_rank = 8`（pos 2 甚至 rank 891）。
**这解释了为什么模型会煞有介事地说"'张量'并非标准术语"**——
不是我们的量化/框架把它弄坏了，这份 checkpoint 本身在这个难度上就是如此。

→ **对路线的影响：`g64 → g32` 的优先级下调**。它最多追回这 ~0.05–0.26 nats，
而代价是 scale 翻倍多吃 ~1.8 GiB 显存（可能被逼回 `LPC=44`，TPOT +16%）。
**速度和长 prompt 才是当前的真瓶颈。**

---


## 16. ★prefill 瓶颈评估：为什么慢、能不能靠 chunked prefill 救

### 16.1 成本拆分（实测，`tests/t_prefill_split.py`）

monkeypatch 掉 attention / MoE 分别计时（eager 48 层，基线 **396.8 ms/token**）：

| 组件 | 时间 | 占比 |
|---|---|---|
| attention（`_deltanet` / `_qsa`） | 125.6 ms | **31.7%** |
| MoE（`_moe`） | 153.5 ms | **38.7%** |
| 其它（mixer/inject/中转/head） | 117.7 ms | 29.6% |

再单独量"地板"（每个 token 都要重做一遍的固定成本）：

| 项 | 实测 | 换算 |
|---|---|---|
| 单层 18 个 int8 稠密权重的 `E.lin`（含反量化） | 3.05 ms/层 | **× 48 = 146.2 ms（占基线 36.9%）** |
| 专家 top-10 两次 `ExpQ.matmul` | 1.53 ms/层 | × 48 = 73.5 ms |
| `lm_head`（606 MiB int8，分块 16384） | **7.51 ms/token** | 只 1 次 |
| 跨卡中转 `h1→host→h1` | 0.10 ms | 可忽略 |

单层稠密权重大户：`w_qkv` 25 MiB、`w_z` 15、`w_out` 15，加其余共 **≈ 73 MiB/层**
（与 §11 的"专家占 94%、dense 77 MiB"互相印证）。
换算到生产口径（捕获后 142.9 ms）：**attention ≈ 45 ms / MoE ≈ 55 ms / 其它 ≈ 42 ms**。

### 16.2 结论：这是**权重带宽 + 每 token 重做反量化**的问题，不是算力问题

- 每 token 要读的权重：稠密 bf16（反量化后）≈ 146 MiB × 48 = **7 GiB**，加专家解包 ≈ 1.1 GiB；
- 真正的算力需求只有 ~100 MFLOP/层 → 理想值 ~100 µs，而实际 143 ms，**差三个数量级**；
- 这也解释了为什么 **prefill 和 decode 一样贵**（134 vs 143 ms/token）：两者都是 B=1 的 GEMV，
  每一步都把整层权重从头读一遍。

### 16.3 三条路的对比

| 方案 | 理论收益 | 代价 / 风险 | 建议 |
|---|---|---|---|
| **A. chunked prefill**（一次算 B=8–32 个 token） | 权重读入摊薄 B 倍 → **3–7×** | 要写 ΔNet 的分块矩阵形式 + 分块 attention + **grouped MoE**（B 个 token 的专家并集）；求和顺序变了，必须重验 `/fnx/loss`；NPUGraph 只能按**固定 B** 另捕一份图 + 尾部补算；dev0 只剩 ~900 MiB 图池 | **暂不做**（数天，研究级） |
| **B. 多请求 batch**（B 条独立序列） | 吞吐 ×2–4 | 每条序列数学不变（好验），但要 batch 状态/KV + 重捕图；**不改善单条长 prompt 延迟** | 有并发需求再做 |
| **C. 进度 + 预估** | 感知延迟大改 | 极小 | ✅ **本轮已做** |
| D. dense 预反量化常驻 | 省掉每 token 反量化 | 73 MiB/层 int8 → 146 MiB bf16，48 层要 +3.4 GiB，**装不下** | 不可行 |
| E. MoE 自定义 kernel | 未验证 | 同布点 `mean_nll` 差 8.6 nats，根因未定位 | 不做 |

### 16.4 已落地的缓解（本轮）

- **streaming 进度**：prefill 按 1/8 刻度发 SSE 注释行 `: prefill i/N`
  （客户端可显示"正在读 prompt"，也顺带避免连接空闲超时）。实测 180-token prompt：
  `3.08s 22/180 → 6.24s 44/180 → … → 24.51s 180/180`，首 token **24.66 s**，速率 **136 ms/token**。
- **非流式预估**：`fnx_stats.est_prefill_s = n_prefill × 0.143`（实测 47 步 → 预估 6.72 s；
  总耗时 9.13 s，其中 24 个输出 token 约 2.4 s，对得上）。

### 16.5 若将来真做 chunked prefill，验收门槛先定好

1. 同一 prompt 的 logits 与逐 token 路径 **maxdiff ≤ 1e-3**（bf16 求和顺序不同，不能要求逐位）；
2. `/fnx/loss`(200) 与 2.1756 的差 **≤ 0.05 nats**；
3. B=16 时 prefill **≥ 2.5×** 提速（否则不值得）；
4. 显存：B=16 的新图池必须塞进 dev0 剩余 ~900 MiB（可能得放 dev1 或与 B=1 图不共存）。

---

## 17. 本轮新增/改动清单

| 文件 | 改动 |
|---|---|
| `fnx/model.py` | `save_state` / `load_state` / `save_into` / `state_to_host` / `state_bytes`；head/mixer 可放 dev1（`FNX_HEAD_DEV`） |
| `fnx/server.py` | `FNX_PROF` 分段计时、`GET /fnx/prof`、`_build_prefix`/`_prefix_hit`/`_conv_store`（LRU 4 槽）、`_tpl_kw`（模板变体）、**停止符修正**、**`/metrics`**、**生成放线程（`asyncio.to_thread`）**、**流式全程持锁** |
| `tests/t_profile_req.py` | 复刻 `generate()` 逐段计时 |
| `tests/t_prefix_cache.py` | 前缀缓存等价性 + 收益 |
| `tests/t_head_ab.py` | ★int8 head vs bf16 head 的 A/B（已证否）+ 定点探针 |
| `tests/t_ref_eval.py` | ★参考 bf16 在同一文本上的逐 token NLL（与线上配对比较） |
| `tools/tok_prefix.py` / `tok_effort.py` / `tok_moe.py` | 模板 token 明细 / effort 开关 / `MoE` 掉字排查 |
| `tools/turn_diag.py` | 多轮前缀公共长度定位（第 2 轮 miss 的根因） |
| `tools/lqp_breakdown.py` | 每层权重构成：**94% 是专家**，dense 只有 77 MiB |
| `tools/mcq_eval.py` | ★24 题单选题任务级评测（准确率 24/24） |
| `tools/compare_ref.py` | ★bf16 参考 vs 线上的逐 token 配对比较 |
| `tests/t_prefill_split.py` | ★prefill 成本拆分（attention/MoE/反量化地板） |
| `backend/remote/p275_prod.sh` | ★生产启动脚本 v5（**48 层 / 46 捕获 / head 在 dev1**，防重启打架） |
| `backend/remote/watchdog.sh` | ★看护：宽限期 + 进程存活判断 + 锁 |
| `backend/remote/p290_systemd.sh` | ★systemd 单元（开机自启 + 看护作为主进程） |
| `backend/remote/p284` / `p285` / `p286` / `p287` / `p288` / `p289` | LPC 扫描 / head 搬 dev1 / 20 请求压测 / 容器主进程化（回滚）/ 并发正确性 / 缓存不变性 |
| `backend/remote/p266…p287` | 本轮全部实测脚本 |

