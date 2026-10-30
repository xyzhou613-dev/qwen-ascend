# qwen-ascend

**Qwen3.8-Flash-Next（W4 MoE，约 121 G 参数）在 2×Ascend 910B4 上的自研推理后端。**

不依赖 vLLM / MindIE / ATB 图引擎：手写 model forward + `torch_npu` NPUGraph 捕获 + 自研 W4 打包/解包，
在**单机双卡 128 GiB 显存**里跑完 48 层，OpenAI 兼容 HTTP 服务。

> **当前状态：基线 `baseline-r33`**
> `mean_nll = 2.1860`（ppl 8.90）/ `top1 = 48.0%` / **TPOT 117–119 ms（8.5 tok/s）**
> TTFT：热态 **0.14 s**，冷启动 ≈ **0.115 s × prompt token 数**
> 24 题单选题准确率 **24/24**，冷启动 **31 s**，错误 0
>
> 提速主线（第 32 轮）：**装载期预反量化** —— 把"每 token 重做一遍"的 int8→bf16 反量化
> 改成装载期一次做完（`FNX_DEQ_SMALL/HEAD/ATTN`），再把专家 **scale 组 g64 → g128** 减半
> 腾出 0.82 GiB 显存。TPOT 139.5 → **118 ms（−15.4%）**，代价是 `mean_nll` +0.010、
> `top1` −1 点（任务级 MCQ 仍 24/24）。
> 详见 `docs/Qwen3.8-Flash-Next-第32轮实测记录.md`。

---

## 1. 这套东西是怎么跑起来的（一句话版）

一台机器 2 张 910B4，各 64 GiB 显存。**48 层全部要放进去**，但 48 × 1.21 GiB = 58 GiB 专家权重
+ 稠密权重 + KV/递归状态 ≈ 62.9 GiB > 单卡可用 60.96 GiB。所以：

```
dev0（60 GiB 上限）              dev1
├─ 46 层图捕获（NPUGraph）       ├─ 2 层 eager（dev1 的图捕获被实验证否）
├─ KV/DeltaNet 递归状态          ├─ lm_head + mixer（int8，646 MiB）
└─ ...                          └─ logits
        每天 token：dev0 跑完 46 层 → h 经宿主中转 → dev1 跑 2 层 + head
```

三个关键的、**反直觉但必须遵守**的设计约束（都有实测记录，见 `docs/`）：

1. **dev1 上 NPUGraph 完全不可用** —— 连 `out = x*2+1` 都捕获不了（replay 后输出仍为零）。
   所以图捕获只能在 dev0，层数分配（`FNX_LPC`）由图捕获容量决定。
2. **专家权重必须 4-bit 打包（LQP artifact），int8 容器不能进显存**，且解包**只能用 bf16 算术**
   （NPU 上整数位运算慢 7–18 ms/16.4M 元素）。每层设备占用 2.49 → **1.21 GiB**。
3. **停止符是 `<|im_end|>`(248046)，不是 pad**；`enable_thinking=False` 让同样问题快 2.25 倍且输出更干净。

## 2. 目录结构

```
fnx/           推理引擎与服务（engine / model / load / server）
tests/         全部实测脚本（深度曲线、捕获证否、前缀缓存等价性、批量微基准…）
tools/         离线工具（LQP 打包、等价性门禁、模板探测、MCQ 评测、参考对比…）
ops/
  registry.yaml  16 个算子的实现状态与已知问题
  scripts/       生产启动、看护、systemd、各轮测量驱动
docs/          交接文档 + 第 29/30/31 轮完整实测记录 + 部署方案
scripts/       ★一键部署与验收（新机器复现用）
```

## 3. 在新机器上复现

前提：**一台与基线同规格的机器**（2×910B4、openEuler 24.03、容器镜像
`quay.io/ascend/vllm-ascend:v0.23.0`、权重放在 `/models`）。
完整步骤见 `scripts/README.md`，最短路径：

```bash
git clone https://github.com/xyzhou613-dev/qwen-ascend.git && cd qwen-ascend
bash scripts/deploy_all.sh          # 自检 → 容器 → 权重 → 起服务 → 验收
```

验收通过时应打印 `BASELINE OK`，并给出与本 README 顶部一致的数字
（判据见 `BASELINE.md` §4）。

## 4. 服务接口

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/v1/chat/completions` | OpenAI 兼容，支持 `stream`；额外接受 `reasoning_effort` / `enable_thinking` |
| GET | `/health` | 存活探针（生成期间也能立刻返回） |
| GET | `/metrics` | Prometheus 文本：请求数/吞吐/命中率/排队/TPOT |
| GET | `/fnx/prof` | 最后一次请求的分段耗时与缓存命中情况 |
| POST | `/fnx/loss` | **质量口径**：teacher-forcing 的 `mean_nll` / `top1` / `median_rank` |
| POST | `/fnx/selfcheck` | 历史上下文自检（有历史 vs 无历史的 NLL） |

响应里的 `fnx_stats` 会带 `n_prefill` / `est_prefill_s`；流式过程中会发 SSE 注释行
`: prefill i/N`，客户端可以据此显示"正在读 prompt"。

## 5. 已知限制（都是实测过的，不是猜测）

| 限制 | 数字 | 记录 |
|---|---|---|
| **prefill 与 decode 一样贵**（每 token 都是 B=1 的 GEMV，要重读整层权重） | 冷启动 TTFT ≈ 0.14 s × prompt token | 第 30 轮 §16 |
| 长 prompt 代价线性 | 300 token ≈ 42 s，780 token ≈ 104 s | 第 30 轮 §16 |
| chunked prefill（能到 3–7×）不做 | 需要 ΔNet 分块矩阵 + grouped MoE，数天工作量、要重验数值 | 第 30 轮 §16.3 |
| 多请求 batching 收益有限 | B=2 只有 1.43×，且单请求 +38%；B=4 在 46 层捕获下 OOM | 第 31 轮 |
| 模型本身在难文本上就弱 | 参考 bf16 同文本 `top1` 也只有 33%；量化只贡献中位 +0.05 nats | 第 30 轮 §15 |
| 单卡容量决定层数上限 | 46 层捕获（59.54 GiB）；`LPC=47` 只差 256 MiB | 第 30 轮 §12 |

## 6. 引用与致谢

- 模型权重：Qwen3.8-Flash-Next（本仓库不含权重）
- 运行时：`torch_npu` / CANN（版本见 `BASELINE.md`）
- 所有性能数字均在 Ascend 910B4 上实测，测量脚本同仓库可复跑
