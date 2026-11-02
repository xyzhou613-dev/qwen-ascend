# LiveCodeBench 评测工具（本仓库版本）

这套脚本用于 `docs/Qwen3.8-Flash-Next-2x910B4-LiveCodeBench100题评测报告.md` 里的全部数字：
100 题抽样、真实执行判题、pass@1 与吞吐统计。数据文件（`lcb_sample100.jsonl` 841 MB、
`sample100_easy.jsonl` 4.8 MB）不入库，见 §4。

## 1. 三个 harness 变体的区别

| 文件 | 与原版差异 | 用途 |
|---|---|---|
| `run_eval.py` | 原版（只把 `import requests` 换成同目录下的 `requests.py` 标准库替身，用于无外网/无 requests 的机器） | 基线评测 |
| `run_eval_fix.py` | 在上面基础上只改 `build_messages` 的一个分支：题目有 `func_name` 但**没有** `starter_code` 时，提示词要求"写成名为 `<func_name>` 的函数、不要读 stdin" | **MBPP+ 正确口径**（原版会退化成 stdin 提示词，而判题器按脚本调用具名函数 ⇒ 必然 NameError） |
| `run_eval_vllm.py` | 在 `run_eval_fix.py` 基础上支持 config 里的 `chat_template_kwargs` | 给 vLLM 部署的模型关思考链（与 Flash-Next 的 `FNX_ENABLE_THINKING=0` 对齐） |

判题器 `testing_util.py` 是**修正版**：比原版多了 `grade_script`，支持 `testtype="script"`
（humanevalplus / mbppplus 的官方测试脚本）。

## 2. 两个必须知道的环境坑

1. **宿主机 python3 没有 numpy**：`testtype=script` 的官方测试脚本要 `import numpy`，
   在裸 python3 上判会全部 `runtime`。⇒ 判题必须用容器内 venv（`/root/ascend-dev/venv/bin/python`）。
2. **判题不能用进程池**：`subprocess.Popen(preexec_fn=…)` 在 `ProcessPoolExecutor` 的 fork 子进程里
   会假失败（同一题进程内判是 `True`，经池变 `runtime`）。⇒ 用 `rejudge2.py`（串行进程内）。
   另注意 config 里 `eval_workers` 对这套判题器无效，保持 1。

## 3. 用法

```bash
# 生成 + 判题（在容器内跑，venv 有 numpy）
docker exec fnx bash -lc "cd /models/eval && /root/ascend-dev/venv/bin/python -u \
  scripts/run_eval_fix.py --config config_easy_910.json \
  --dataset dataset/sample100_easy.jsonl --out results/easyfix_run1"

# 只重新判题（已生成过 generations.jsonl 时用，串行）
docker exec fnx bash -lc "cd /models/eval && /root/ascend-dev/venv/bin/python \
  rejudge2.py results/easyfix_run1 dataset/sample100_easy.jsonl"

# 汇总多轮
python3 aggregate.py
```

配置（`configs/`，都是 `api_key: EMPTY` 的模板）：

| 文件 | 用途 |
|---|---|
| `config_910.json` | 910 上的自研 fnx 引擎，`max_tokens=2048`（混合套 A 套口径） |
| `config_easy_910.json` | 910 上的 fnx 引擎，`max_tokens=32768`（全 easy 的 B 套，与 201 同口径） |
| `config_hard32768.json` | 910 上的 fnx 引擎，混合套 A 套 + 32768（不截断） |
| `config_27b.json` | vLLM 部署的 Qwen3.8-27B，`max_tokens=16384` + 关思考链 |

## 4. 数据文件

| 文件 | 大小 | 来源 |
|---|---|---|
| `lcb_sample100.jsonl` | 841 MB | 用户提供的 `flash_next_sample/dataset/`，LiveCodeBench release_v1~v6 按平台×难度分层抽样 100 题（2341 测试点） |
| `sample100_easy.jsonl` | 4.8 MB | 100 题全 easy（humanevalplus 33、mbppplus 33、atcoder 15、leetcode 15、codeforces 4；742 测试点） |

两者都**不入库**（体积），部署时放在 `/models/eval/dataset/`。

## 5. 产出

`results/<轮次>/summary.json`（pass@1、分平台、逐题明细）、`generations.jsonl`（原始输出 + 抽取的代码）、
`summary_rejudged.json`（用修正判题器重判的结果）。
