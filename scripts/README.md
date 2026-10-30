# 部署脚本说明（在新机器上复现基线 `baseline-r31`）

目标机要求：**与基线同规格**——2 × Ascend 910B4（每卡 64 GiB）、openEuler 24.03、
CANN 9.1.0 + 驱动 26.1.1、docker 可用、能拉到镜像 `quay.io/ascend/vllm-ascend:v0.23.0`、
权重放在 `/models`（见 `BASELINE.md` §2）。

## 一条命令

```bash
git clone https://github.com/xyzhou613-dev/qwen-ascend.git
cd qwen-ascend
bash scripts/deploy_all.sh
```

它会依次做：

| 步骤 | 脚本 | 做什么 |
|---|---|---|
| 00 | `00_precheck.sh` | 只读自检：卡/设备节点/docker/镜像/权重/磁盘/评测文本，逐条 OK/FAIL |
| 10 | `10_env.sh` | 建或复用容器 `fnx`（挂 `/models`、`/root/ascend-dev`、5 个 davinci 设备、`--network host`）；建 venv 并装依赖；把代码拷到 `/root/ascend-dev/fnx` |
| 20 | `20_weights.sh` | 有 48 层 LQP artifact 就用；没有就 `tools/repack_v2.py` 生成（约 860 s）+ `tools/check_lqp.py` 等价性门禁 |
| 30 | `30_deploy.sh` | 写 `/root/ascend-dev/p275.sh` + `watchdog.sh`，安装 systemd 单元 `fnx.service`（开机自启），启动并等 `/health` |
| 40 | `40_verify.sh` | 对照 `BASELINE.md` §4 验收，全部通过打印 **BASELINE OK** |

常用开关：

```bash
bash scripts/deploy_all.sh --skip-lqp   # 不建 LQP（服务退回运行时打包，冷启动 ~800 s）
bash scripts/deploy_all.sh --quick      # 验收跳过 24 题 MCQ（约省 4 分钟）
bash scripts/40_verify.sh               # 只跑验收（服务已在跑）
```

可覆盖的环境变量：`RUN`（默认 `/root/ascend-dev`）、`CTN`（`fnx`）、`MDL`（`/models`）、
`IMG`、`PIP_INDEX_URL` / `PIP_TRUSTED_HOST`（内网 pip 镜像）。

## 分步手动执行（排查时用）

```bash
bash scripts/00_precheck.sh                 # 1. 先看有什么问题
bash scripts/10_env.sh                      # 2. 环境
bash scripts/20_weights.sh auto             # 3. 权重（skip / build / auto）
bash scripts/30_deploy.sh                   # 4. 起服务
bash scripts/40_verify.sh --quick           # 5. 验收

systemctl status fnx                        # 看服务
systemctl restart fnx                       # 重启
journalctl -u fnx -n 50                     # systemd 视角
tail -50 /root/ascend-dev/logs/server.log   # 引擎视角（上一份在 server.log.prev）
```

## 验收会检查什么

`40_verify.sh` 直接打服务，和 `BASELINE.md` §4 逐项比对（容差见该表）：

- `POST /fnx/loss`（`eval_text` 前 200 token）→ `mean_nll ≈ 2.1756`、`top1 ≈ 49%`
- 96 token 输出的稳态 TPOT（≤ 175 ms/token）
- 重复 prompt 的 TTFT（≤ 0.5 s，应命中前缀/对话缓存）
- `dev0` 显存分配（≤ 60.4 GiB）
- 24 题单选题准确率（≥ 22/24，`--quick` 可跳过）

> **注意**：`mean_nll` 是文本相关的绝对值，必须用仓库 `tests/eval_text.txt`
> （部署时会拷到 `/root/ascend-dev/eval_text.txt`），换文本则数字不可比。

## 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `00` 报"只检测到 0/1 张 910B4" | 卡没被识别：先 `npu-smi info` 确认驱动与容器设备挂载 |
| `10` 报 venv 创建失败 | 机器无外网：给 `PIP_INDEX_URL`（如内网源），或手工把 venv 拷过来 |
| `30` 卡在 `图捕获` / OOM | 显存不足：确认 `FNX_LPC=46`、没有别的进程占卡（`npu-smi info` 看 Process） |
| `40` 的 `mean_nll` 明显偏高（如 3.18） | 层数没跑满 48（检查 `FNX_NLAYERS`），或 LQP 缺失导致退化配置 |
| `40` TPOT 偏高 | 是否有人跑了大任务占 CPU；或 `FNX_LAYERS`/`LPC` 被改成 44（会退到 166 ms） |
| 服务反复重启 | 看 `logs/watchdog.log`：正常情况不应出现"触发重启"；出现说明 health 不通 |

## 把改动推回 GitHub（需要你自己的凭据）

仓库里没有、也不应该有任何凭据。首次推送时用 **Personal Access Token（fine-grained，只给该仓库
`Contents: Read and write`）**：

```bash
cd qwen-ascend
git remote add origin https://github.com/xyzhou613-dev/qwen-ascend.git
git add -A && git commit -m "..."        # 若已有改动
git tag -a baseline-rNN -m "..."         # 需要新基线时
GIT_ASKPASS=/bin/true \
  git -c credential.helper= \
      push "https://x-access-token:<YOUR_TOKEN>@github.com/xyzhou613-dev/qwen-ascend.git" \
      HEAD:main --tags
```

> 推送后请立即 `git remote set-url origin https://github.com/xyzhou613-dev/qwen-ascend.git`
> 并清掉 shell 历史里的 token，避免凭据留痕。
