> 2026-11-02：ACL设备IPC混合传输已验收，当前hybrid；256k生产TTFT 102.114/101.607s。[第91–92轮实测与边界](docs/r92-peer-ipc.md)。

> 2026-11-01 本轮两次复测：256k TTFT 103.629/102.921 秒，输出 18.45/18.46 tokens/s；[本轮实验与部署](docs/r87-production.md)。

> 2026-11-01 更新：默认部署 256k（24+24 独立进程），新文档 262016 输入 + 128 输出，
> 本机 HTTP TTFT 103.956 s，输出 18.56 tokens/s，三位置检索 3/3。
> 详见 [生产验收与各轮记录](docs/r80-production.md)。

# qwen-ascend

**Qwen3.8-Flash-Next（W4 MoE，约 121 G 参数）在 2×Ascend 910B4 上的自研推理后端。**

不依赖 vLLM / MindIE / ATB 图引擎：手写 model forward + `torch_npu` NPUGraph 捕获 + 自研 W4 打包/解包，
在**单机双卡 128 GiB 显存**里跑完 48 层，OpenAI 兼容 HTTP 服务。

> **当前状态：基线 `baseline-r55`**（现网部署：双进程流水 + 分块图 prefill + 稀疏 QSA + grouped MoE + 256k 上下文）
> **256k 新 prompt**：prefill **2,246.9 tok/s**、TTFT **117.86 s**、decode **55.97 ms/token（17.87 tok/s）**、长文三事实检索 **3/3**
> **8k 新 prompt**：TTFT **6.99 s**、端到端 14.1 s、TPOT 55.96 ms
> 短请求：TPOT **54.1 ms（18.5 tok/s）**（本仓库独立实测）；`mean_nll 2.1653`（基线 2.1648）、MCQ **24/24（0.52 s/题）**
> 显存：dev0 **56.0 GB** / dev1 **56.6 GB**（两卡均衡），`FNX_CTX=262144`、48 层按 24+24 切分
>
> 提速主线（第 37/38 轮）：**把整数权重的反量化全部搬进 Cube** —— 用 CANN 现成的
> `torch_npu.npu_weight_quant_batchmatmul`（aclnn per-group）替掉"每 token 反量化 + bf16 GEMM"：
> 专家段（W4A16）一层 1.0 → 0.32 ms，稠密段（W8）一层 0.953 → 0.263 ms，
> 再把共用输入的投影（qkv+z / b+a / mixer down+inj / shared gate+up）拼成一次调用。TPOT 110.3 → **65.8 ms（−40%）**，prefill 105 → **67.7 ms/token**，
> `mean_nll` 从 2.1756 → r38 的 **2.1665**（中间不再落 bf16，累加更准）→ r39 的 2.1735（拼接改变 GEMM 累加顺序），MCQ 仍 24/24。
> 四个开关：`FNX_MOE_W4A16` / `FNX_DENSE_W8` / `FNX_DENSE_CAT` / `FNX_SMALL_MERGE`。详见 `docs/…第37/38/39轮实测记录.md`。
>
> **第 49–55 轮（外部并入，现网版本）**：① 分块 + **分块图捕获** prefill（r49，2.1×）；
> ② **expert-major 重排 + GroupedMatmulV5** 的 grouped MoE + FP32 分块 DeltaNet（r50，prefill 24.8 → **558.6 tok/s**）；
> ③ **稀疏 QSA indexer + BF16 连续 KV** 打开长上下文（r52/53，8k → 256k）；
> ④ **每卡独立进程、24+24 流水**（r54/55，256k TTFT **117.9 s**、decode **55.97 ms**）。
> 架构总览与"哪些 r48 结论被推翻"见技术文档 **§4.22**；各轮细节见 `docs/…第49 … 第54-55轮…`。
>
> **已知边界**：256k 是单次新 prompt 的实测，不代表所有输入/并发下的延迟保证；decode 的 54–56 ms/token 仍是最大项
> （expert-major 只用于 prefill）；`FNX_BATCH=1`，未做并发批量。

>
> 提速主线（第 37/38 轮）：**把整数权重的反量化全部搬进 Cube** —— 用 CANN 现成的
> `torch_npu.npu_weight_quant_batchmatmul`（aclnn per-group）替掉"每 token 反量化 + bf16 GEMM"：
> 专家段（W4A16）一层 1.0 → 0.32 ms，稠密段（W8）一层 0.953 → 0.263 ms，
> 再把共用输入的投影（qkv+z / b+a / mixer down+inj / shared gate+up）拼成一次调用。TPOT 110.3 → **65.8 ms（−40%）**，prefill 105 → **67.7 ms/token**，
> `mean_nll` 从 2.1756 → r38 的 **2.1665**（中间不再落 bf16，累加更准）→ r39 的 2.1735（拼接改变 GEMM 累加顺序），MCQ 仍 24/24。
> 四个开关：`FNX_MOE_W4A16` / `FNX_DENSE_W8` / `FNX_DENSE_CAT` / `FNX_SMALL_MERGE`。详见 `docs/…第37/38/39轮实测记录.md`。
>
> **第 48 轮（分块 prefill，已被第 49–55 轮接续）**：`FnLayer.forward_chunk` + `FnModel.prefill_chunk`（`FNX_CHUNK>1`），
> 与当时的生产逐 token 路径**逐位等价**（`tests/t_chunk_state.py` 全部 `0.000e+00`）。当时实测稠密+注意力+递推摊薄 **8.6×**
> （156 → 18.1 ms/token，T=32 饱和），但 MoE 不摊薄 ⇒ 无图版 chunk=128 的 75.8 ms/token 比当时的 61.7 慢 1.23×。
> **该"负收益"结论已在 r49 被修正**：补上分块图捕获后分块 prefill 转为正收益（2.1×，r49），
> 分组 MoE 在改用 GroupedMatmulV5 后也是正收益（r50）。详见 `docs/…第48轮实测记录.md`、§4.21 与 §4.22。


---

## 1. 这套东西是怎么跑起来的（一句话版）

一台机器 2 张 910B4，各 64 GiB 显存。**48 层全部要放进去**，但 48 × 1.21 GiB = 58 GiB 专家权重
+ 稠密权重 + KV/递归状态 ≈ 62.9 GiB > 单卡可用 60.96 GiB。所以：

```
dev0（60 GiB 上限）              dev1
├─ 45 层图捕获（NPUGraph）       ├─ 3 层 eager（dev1 的图捕获被实验证否）
├─ KV/DeltaNet 递归状态          ├─ lm_head + mixer（bf16 预反量化）
└─ ...                          └─ logits
        每天 token：dev0 跑完 45 层 → h 经宿主中转 → dev1 跑 3 层 + head
```

三个关键的、**反直觉但必须遵守**的设计约束（都有实测记录，见 `docs/`）：

1. **dev1 上 NPUGraph 完全不可用** —— 连 `out = x*2+1` 都捕获不了（replay 后输出仍为零）。
   所以图捕获只能在 dev0，层数分配（`FNX_LPC`）由图捕获容量决定。
2. **专家权重必须 4-bit 打包（LQP artifact），int8 容器不能进显存**；每层设备占用 2.49 → **1.21 GiB**。
   **r37 起解包不再由 torch 做**：走 `npu_weight_quant_batchmatmul`（Cube 融合 per-group W4A16），
   artifact 由 `tools/repack_w4a16.py` 生成（int4pack 布局，同大小），每层专家段 1.0 → 0.32 ms。
3. **停止符是 `<|im_end|>`(248046)，不是 pad**；`enable_thinking=False` 让同样问题快 2.25 倍且输出更干净。

## 2. 目录结构

```
fnx/           推理引擎与服务（engine / model / load / server）
tests/         全部实测脚本（深度曲线、捕获证否、前缀缓存等价性、批量微基准、W4A16 一致性…）
tools/         离线工具（LQP 打包、int4pack 重打包、等价性门禁、模板探测、MCQ 评测、参考对比…）
ops/
  registry.yaml  16 个算子的实现状态与已知问题
  scripts/       生产启动、看护、systemd、各轮测量驱动
docs/          交接文档 + 第 29/30/31/32/37 轮完整实测记录 + 部署方案
scripts/       ★一键部署与验收（新机器复现用）
```

> **r37 额外的前置步骤**：`docker exec fnx bash -lc 'cd /root/ascend-dev/fnx && \
> /root/ascend-dev/venv/bin/python tools/repack_w4a16.py 48'`（一次，约 22 分钟），
> 产出 `/models/Qwen3.8-Flash-Next-LQP-w4a16`。没有它时 `FNX_MOE_W4A16=1` 会静默回落到旧的 v2 路径。

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
