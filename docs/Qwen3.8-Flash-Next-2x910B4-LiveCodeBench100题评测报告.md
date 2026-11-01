# Qwen3.8-Flash-Next 代码能力评测：910B4（W4）vs RTX PRO 6000（IQ3_S）

日期：2026-11-01
被测两端：

| 端 | 机器 | 部署 | 量化 | 上下文 | 实测输出速度 |
|---|---|---|---|---|---|
| **910** | 192.168.18.206，2×Ascend 910B4（128 GiB） | 自研 fnx 引擎（r55：双进程流水 24+24、分块图 prefill、稀疏 QSA） | **W4**（int4 每 64 组）+ `FNX_NO_PLE=1` | 262,144 | **18.9 tok/s** |
| **201** | 192.168.19.201，RTX PRO 6000 Blackwell 96 GB | Strata 0.2.3（docker，engine 0.1.48） | **IQ3_S**（~3 bit） | 262,144 | **~220 tok/s** |

两套题（同一份文件，md5 校验一致，两台机器都跑）：

- **A 套（难易混合）** `lcb_sample100.jsonl`：LiveCodeBench 分层抽样 100 题 / 2341 测试点
  （atcoder 51、leetcode 40、codeforces 9；easy 34 / medium 33 / hard 33）
- **B 套（全 easy）** `sample100_easy.jsonl`：100 题 / 742 测试点
  （humanevalplus 33、mbppplus 33、atcoder 15、leetcode 15、codeforces 4）

## 1. 结论摘要

| 端 | A 套（混合，2048 上限） | B 套（全 easy，32768 上限） | 备注 |
|---|---|---|---|
| **910（W4）** | **64/100（64.0%）** — 2 轮完全一致 | **67/100（67.0%）** — 3 轮完全一致 | A 套 29/100 撞 2048 上限 |
| **201（IQ3_S）** | 96/96/95（另一次会话，32768 上限） | **91/91/91（91.0%）** | 无截断 |
| 同一口径可比项 | 口径不同，只作参考 | **67 vs 91** | 差距**全部在 MBPP+** |

**核心发现：B 套上 910 与 201 的 24 分差距，100% 来自 `mbppplus`（1/33 vs 26/33）；
其余四个来源两边持平甚至 910 更好**：

| 来源 | 910（W4） | 201（IQ3_S） | |
|---|---:|---:|---|
| atcoder | 15/15 | 15/15 | 平 |
| codeforces | 4/4 | 4/4 | 平 |
| leetcode | 15/15 | 15/15 | 平 |
| humanevalplus | **32/33** | 31/33 | 910 略好 |
| **mbppplus** | **1/33** | 26/33 | **唯一分水岭** |

`humanevalplus` 与 `mbppplus` 判题方式相同（都是 `testtype=script`，把官方测试脚本和模型代码拼起来跑），
所以差异不是判题器造成的 —— 是**题面协议**造成的，见 §3。

## 2. 结果明细

### 2.1 910 侧（本次实测，逐轮独立）

| 轮次 | A 套（2048） | B 套（32768） | 生成时长 | 判题时长 |
|---|---|---:|---:|---:|
| A-run1 | 64/100 | — | 94.5 min | 14.8 min |
| A-run2 | 64/100 | — | 94.5 min | 14.8 min |
| B-run1 | — | 67/100 | 32.5 min | 0.1 min* |
| B-run2 | — | 67/100 | 32.5 min | — |
| B-run3 | — | 67/100 | 32.5 min | — |

\* B 套判题时间短是因为 742 个测试点且其中 66 个（MBPP+/HE+）走脚本判题。

**确定性**：A 套两轮不仅通过题数相同（64），连 `completion_tokens` 总数（107,181）都完全相同；
B 套三轮逐题结果完全一致。⇒ 该引擎在 greedy 采样下是**确定性**的，多轮跑的价值在于排除抖动。

**A 套分平台/难度**（两轮一致）：

| 平台 | 通过/总 | | 难度 | 通过/总 |
|---|---:|---|---|---:|
| atcoder | 28/51 | | easy | **34/34** |
| codeforces | 8/9 | | medium | 22/33 |
| leetcode | 28/40 | | hard | 8/33 |

**A 套截断**：29/100 题正好停在 `max_tokens=2048`（`completion_tokens >= 2048`），
且全部集中在 hard（8/33 通过）与偏难的 medium；easy 题 34/34 通过、无一截断。
⇒ **64% 是"2048 token 预算"下的下界**，不是模型能力上界。

### 2.2 201 侧（对照，32768 上限）

| 轮次 | A 套 | B 套 |
|---|---:|---:|
| run1 | 96/100 | 91/100 |
| run2 | 96/100 | 91/100 |
| run3 | 95/100 | 91/100 |

A 套（10-31 21:09 起）与 B 套（11-01 10:43 起）是两次不同的会话，A 套的 9 轮数据其一。
B 套三份 `summary.json` 的逐题结果完全相同。

## 3. 关键发现：MBPP+ 的题面与判题器不一致（不是模型能力差异）

B 套里 MBPP+ 的题 **`starter_code` 为空**，于是本 harness 的 `build_messages` 走"stdin 模式"，提示词写成：

> `### Format: Read the inputs from stdin solve the problem and write the answer to stdout`

模型据此写了 stdin 程序。但判题器按 `testtype=script` 执行官方测试脚本，测试脚本里直接调用**期望的函数名**：

```text
mbpp/397  期望函数名 median_numbers
910 生成：def find_median(a, b, c) + def main(): sys.stdin.read() ...
判题：NameError: name 'median_numbers' is not defined
```

`mbpp/94` 同理：模型写了一整个 `main()`，期望的是 `index_minimum`。

所以 MBPP+ 这 33 题实际测的是"模型能不能猜中函数名"，而不是编码能力：

- 910（W4）严格照提示词写了 stdin 程序 ⇒ 1/33；
- 201（IQ3_S）无视提示词、直接写同名函数 ⇒ 26/33。

**这是题面缺陷，应当在 prompt 里给出函数签名**（例如把 `starter_code` 合成为
`def <func_name>(...):`）再比。这也是为什么 `humanevalplus`（题面自带完整签名）两边几乎满分
且 910 还更高：`32/33` vs `31/33`。

## 4. 吞吐与成本

| 项 | 910 | 201 | 倍数 |
|---|---:|---:|---:|
| 输出速度（稳态） | 18.9 tok/s | ~220 tok/s | **11.6×** |
| A 套单轮生成 | 94.5 min（107k token） | ~10 min（~125k token） | ~9–10× |
| B 套单轮生成 | 32.5 min（36k token） | ~9 min（~125k token） | — |
| 首 token（15 token prompt） | 0.47 s | 9.9 s（冷）/0.4 s（热） | — |

> B 套 910 的 wall time 只慢 3.4× 而不是 11.6×，是因为它在 MBPP+ 上写的是短 stdin 程序（36k token），
> 而 201 写了 125k token。**同样的输出量下，速度差就是 11.6 倍。**

## 5. 局限

1. **两套题都只有 100 题**，A 套分层抽样自 1055 题，统计误差约 ±5 个百分点（95% CI）。
2. **A 套口径不同**：910 用 2048 上限（29 题截断），201 用 32768，不能直接排座次；
   要严格对比需要 910 也跑 32768（预计 ~6–8 h/轮，因为难题会写很长）。
3. **B 套的 MBPP+ 结论不成立**（见 §3），要等修正 prompt 后重跑才算数。
4. **两侧量化不同**（W4 int4 per-64-group vs IQ3_S ~3 bit），且 910 关了 PLE；
   本报告不能把差异归因于"卡"或"量化"其中一项。
5. **判题环境差异**：910 的判题必须在容器内跑（宿主机 python3 无 numpy，`testtype=script`
   的官方脚本要 `import numpy`）；另外进程池 fork + `preexec_fn` 会假失败，必须串行判题。
   这两点都是**评测环境**问题，已在 `rejudge2.py` 里修正。

## 6. 复现

```bash
# 910（容器内跑，venv 有 numpy；数据与脚本在 /models/eval）
docker exec fnx bash -lc "cd /models/eval && /root/ascend-dev/venv/bin/python \
    scripts/run_eval.py --config config_easy_910.json \
    --dataset dataset/sample100_easy.jsonl --out results/easy_runX"
# 修正判题（宿主机跑会因缺 numpy 全部 runtime；进程池会假失败）
docker exec fnx bash -lc "cd /models/eval && /root/ascend-dev/venv/bin/python \
    rejudge2.py results/easy_runX dataset/sample100_easy.jsonl"

# 201
cd /home/server/flash_next_sample && python3 scripts/run_eval.py   # 默认即 sample100_easy
```

判题器差异：201 的 `scripts/testing_util.py` 含 `grade_script`（支持 `testtype=script`），
910 上的版本已同步为同一份（md5 `c179b7f4…`）。
