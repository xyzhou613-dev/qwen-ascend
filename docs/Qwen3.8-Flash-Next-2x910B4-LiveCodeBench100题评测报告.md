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

| 端 | A 套（混合，2048 上限） | A 套（混合，32768 上限） | B 套（全 easy，原协议） | **B 套（全 easy，修正协议）** |
|---|---|---|---|---|
| **910（W4）** | **64/100** ×2 轮一致 | 跑中（见 §7） | 67/100 ×3 轮一致 | **89/100** |
| **201（IQ3_S）** | 96/96/95（其自身会话） | — | 91/91/91 | **90/100** |
| 同口径可比 | 口径不同 | — | 协议有缺陷 | **89 vs 90 ≈ 打平** |

**结论一：把题面协议修正后，910B4（W4）与 RTX PRO 6000（IQ3_S）在代码题上基本打平（89 vs 90）。**

**结论二：先前 67 vs 91 的 24 分差距，是题面缺陷造成的假差距，不是模型能力差。** 修正协议后：

| 来源 | 910（W4）原协议 → 修正 | 201（IQ3_S）原协议 → 修正 |
|---|---|---|
| atcoder | 15/15 → 15/15 | 15/15 → 15/15 |
| codeforces | 4/4 → 4/4 | 4/4 → 4/4 |
| leetcode | 15/15 → 15/15 | 15/15 → 15/15 |
| humanevalplus | 32/33 → 32/33 | 31/33 → 30/33 |
| **mbppplus** | **1/33 → 23/33** | 26/33 → 26/33 |
| **合计** | **67 → 89** | **91 → 90** |

修正只改了 MBPP+ 那 33 题的提示词（见 §3），其余题一字未动。

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

## 3. 关键发现：MBPP+ 的题面与判题器不一致（已修正并复测）

B 套里 MBPP+ 的题 **`starter_code` 为空**，于是原版 harness 的 `build_messages` 走"stdin 模式"，提示词写成：

> `### Format: Read the inputs from stdin solve the problem and write the answer to stdout`

模型据此写了 stdin 程序。但判题器按 `testtype=script` 执行官方测试脚本，测试脚本里直接调用**期望的函数名**：

```text
mbpp/397  期望函数名 median_numbers
910 生成：def find_median(a, b, c) + def main(): sys.stdin.read() ...
判题：NameError: name 'median_numbers' is not defined
```

`mbpp/94` 同理：模型写了一整个 `main()`，期望的是 `index_minimum`。

所以原协议下 MBPP+ 这 33 题实际测的是"模型能不能猜中函数名"：
910（W4）严格照提示词写 stdin 程序 ⇒ 1/33；201（IQ3_S）无视提示词、直接写同名函数 ⇒ 26/33。

**修正做法**（`scripts/run_eval_fix.py`，只改 `build_messages` 的一个分支）：
当题有 `func_name` 但没有 `starter_code` 时，提示词改为

> `### Format: Write your solution as a Python function named `<fn>` (keep this exact name) ...`
> `Read nothing from stdin; the tests import your code and call `<fn>(...)` directly.`

HumanEval+（自带签名）与 atcoder/codeforces/leetcode（stdin 型）的提示词**一字未改**。

**修正后复测（各 1 轮，同一份数据、同样 `max_tokens=32768`、温度 0）**：

| | 910（W4） | 201（IQ3_S） |
|---|---:|---:|
| mbppplus | **23/33** | 26/33 |
| 合计 | **89/100** | **90/100** |
| 生成 token | **31,024** | 171,830 |
| 生成墙钟 | 28.2 min | 11.7 min |

⇒ 两台机器**基本打平**；差别只在风格：910 用 1/5 的 token 写出同样的通过率，201 写得长但速度是 11.6×。
这也反向证明：**原协议的 67 分不能用来评价这个模型**。

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
2. **A 套的口径仍不对等**：910 的 64 分是 `max_tokens=2048`（29 题截断），201 的 96 分是 32768。
   910 已按你的要求补跑一次 32768（见 §7），完成后这一格才可比。
3. **两侧量化不同**（W4 int4 per-64-group vs IQ3_S ~3 bit），且 910 关了 PLE；
   本报告不能把差异归因于"卡"或"量化"其中一项。
4. **判题环境差异**：910 的判题必须在容器内跑（宿主机 python3 无 numpy，`testtype=script`
   的官方脚本要 `import numpy`）；另外进程池 fork + `preexec_fn` 会假失败，必须串行判题。
   这两点都是**评测环境**问题，已在 `rejudge2.py` 里修正。
5. **修正协议的复测各只有 1 轮**（不是 3 轮）：910 原协议 3 轮逐题完全一致、201 也 3 轮一致，
   可认为 greedy 下结果稳定；若要更严格，可再补 2 轮。

## 7. 进行中：910 的 A 套 32768 上限补跑

为把 A 套对齐到 32768（201 的口径），910 上已排队：用修正协议、`max_tokens=32768`
对 `lcb_sample100`（100 题 / 2341 测试点）跑 1 轮。预计 **6–8 h**
（难题会写很长；2048 上限时 29 题被截断，这些题放宽后单题可达 2–4 min）。

结果目录 `/models/eval/results/hard32768_run1_*/`，日志 `/models/eval/logs/hard32768_run1.log`；
完成后本报告会补上"910@32768"这一格，与 201 的 96/96/95 直接对比。


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
