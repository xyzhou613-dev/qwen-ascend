# 基线 `baseline-r44`（可复现性锚点）

本文件定义"基线"到底是什么：**配置、权重、版本、指标、验收判据**。
在新机器上复现后，`scripts/40_verify.sh` 会把实测值与本文件对比，全部在容差内才打印 `BASELINE OK`。

---

## 1. 配置（等价于环境变量）

```bash
# 模型布局
FNX_LPC=45                 # dev0 图捕获层数（留一层显存给"预反量化"换速度）
FNX_NLAYERS=48             # 总层数（必须 48，少 2 层 mean_nll 从 2.18 退化到 3.19）
FNX_HEAD_DEV=              # 空 = auto：head/lm_head 放 dev1（dev1 显存富余）
# 量化与打包
FNX_PACK=0  FNX_EXP_G=64  FNX_MOE_KERNEL=0  FNX_NO_PLE=1  FNX_PACKED_TORCH=1
FNX_LQP=/models/Qwen3.8-Flash-Next-LQP      # 4-bit 打包 artifact（存在即用）
# r37：专家 GEMM 走 aclnn per-group W4A16（Cube）。=0 回到 r35 口径（110.5 ms / 2.1756）
FNX_MOE_W4A16=1
# r38：稠密 int8 权重也走同一个 aclnn 算子；再把共用输入的投影拼成一次调用
FNX_DENSE_W8=1
FNX_DENSE_CAT=1
# r39：mixer 的 down+inj、shared-expert 的 gate+up 也拼成一次 GEMM（1=都拼, mix/sh=只拚一边, 0=不拼）
FNX_SMALL_MERGE=1
FNX_DN_RMS=1              # r42：DeltaNet 输出归一化用融合 npu_rms_norm（更快且 nll 更低）
FNX_W4A16_DIR=/models/Qwen3.8-Flash-Next-LQP-w4a16   # tools/repack_w4a16.py 产出
# 装载期预反量化（第 32 轮，数值逐位不变，只是不再每 token 重算）
FNX_DEQ_SMALL=1            # 每层 11 个小权重（mixer/shared-expert/gate）→ bf16 常驻
FNX_DEQ_HEAD=1             # lm_head（606 MiB int8）→ bf16 常驻（放在 dev1）
FNX_DEQ_ATTN=0             # r38 起不再需要：稠密投影走 aclnn W8，省 0.55 GiB（r37 及以前是 10）
# 服务行为
FNX_PREFIX_CACHE=1  FNX_CONV_CACHE=4  FNX_ENABLE_THINKING=0  FNX_PROF=0
FNX_PORT=8000  FNX_CTX=2048      # 本机硬上限 16k（32k 启动 OOM），生产固定 2048  FNX_BATCH=1
```

布局：**dev0 = 45 层图捕获（59.57 GiB）**，**dev1 = 3 层 eager + lm_head(bf16)/mixer（5.27 GiB）**。
停止符 = `{248044 <|endoftext|>, 248046 <|im_end|>}`。

## 2. 权重（不在仓库里）

| 目录 | 体积 | 作用 |
|---|---|---|
| `/models/Qwen3.8-Flash-Next` | 336 G | 原始 bf16 checkpoint（tokenizer、config、以及参考实现对照用） |
| `/models/Qwen3.8-Flash-Next-Q` | 178 G | int8 容器（逐层稠密 + 专家 q4/s4） |
| `/models/Qwen3.8-Flash-Next-QP` | 64 G | 4-bit 打包专家（中间产物） |
| `/models/Qwen3.8-Flash-Next-LQP` | 59 G | 4-bit 专家 artifact（v2 布局 + offset-binary + s8/base，48 层）；稠密与 `FNX_MOE_W4A16=0` 时仍读它 |
| `/models/Qwen3.8-Flash-Next-LQP-w4a16` | 58 G | **r37 线上读的专家 artifact**（int4pack `[K,E,N/8]` + `[G,E,N]` s8 + bf16 base），由 `tools/repack_w4a16.py` 生成，约 22 分钟 |

`LQP` 可由 `tools/repack_v2.py` 从 `Q` 目录生成（48 层约 860 s，可续跑，原子替换），
生成后必须跑 `tools/check_lqp.py`（与运行时打包逐位等价，`0.000e+00`）才能用。

## 3. 版本矩阵

| 项 | 版本 |
|---|---|
| OS / 内核 | openEuler 24.03 (LTS-SP4) / 6.6.0-159.4.3.154.oe2403sp4.aarch64 |
| 加速卡 | 2 × Ascend 910B4（910B4-1，每卡 64 GiB HBM） |
| 驱动 / npu-smi | 26.1.1（ascendhal 7.35.23） |
| CANN | 9.1.0（`/usr/local/Ascend/cann-9.1.0`） |
| 容器镜像 | `quay.io/ascend/vllm-ascend:v0.23.0`（`--network host --ipc host`，5 个 davinci 设备） |
| Docker | 18.09.0 |
| Python | 3.12.13 |
| torch / torch_npu | 2.10.0+cpu / 2.10.0.post4 |
| transformers / safetensors | 5.21.0 / 0.8.0 |
| fastapi / uvicorn / numpy | 0.123.10 / 0.52.3 / 1.26.4 |

## 4. 验收判据（`scripts/40_verify.sh` 自动比对）

| 指标 | 基线值 | 容差 |
|---|---|---|
| `/fnx/loss`（`eval_text` 前 200 token）`mean_nll` | **2.1648** | ±0.03（r38 为 2.1665，r35 为 2.1756） |
| 同上 `top1` | 48.00% | ±2.0 个百分点 |
| 稳态 TPOT（≥128 token 输出） | **64.6 ms（15.5 tok/s，样本 64.0–64.9）** | ≤ 75 ms |
| TTFT（热态，重复 prompt） | **0.08 s** | ≤ 0.5 s |
| 冷启动（装配 + 前缀缓存 + 图捕获） | ≈30 s | ≤ 90 s |
| 显存 | dev0 **58.97 GiB** / dev1 **5.3 GiB** | dev0 ≤ 60.4 GiB |
| 24 题单选题准确率 | 24/24 | ≥ 22/24 |

**质量口径注意**：`mean_nll` 用的是 `eval_text.txt`（仓库 `tests/` 里的评测文本，
`scripts/20_weights.sh` 会检查它在 `/root/ascend-dev/eval_text.txt`）。
换文本必须重新标定，否则数字不可比。

## 5. 关键实测结论（复现时可能踩到的坑）

1. **dev1 图捕获不可用**（平凡链都失败）⇒ 图只能捕在 dev0，`FNX_LPC` 受显存限制。
2. **PLE（n-gram 查表）有害** ⇒ `FNX_NO_PLE=1`（关掉后 `mean_nll` 3.76 → 3.18、top1 37% → 48%）。
3. **int8 容器不能进显存**（峰值多 1.56 GiB/层 → 37 层就 OOM）⇒ 宿主侧打包。
4. **解包必须用 bf16 算术**（整数位运算每 16.4M 元素慢 7–18 ms）。
5. **停止符必须是 `<|im_end|>`**；否则模型越过轮尾继续编造下一轮模板。
6. **`async def` 里做同步生成会堵死事件循环** ⇒ 生成必须放线程（否则看护会误判并重启）。
7. **`enable_thinking=False`**：同一组问题 50.7 s → 22.6 s，且不再漏 `</think>`。
8. **aclnn 的 `antiquant_scale` 必须与 `x` 同 dtype**（bf16/bf16）；传 fp16 会报 `161002`，极易误读成"不支持 per-group"（第 37 轮 §1）。
9. **流式请求必须读满 `[DONE]`**：客户端中途断开会让服务端卡死（`/health` 也阻塞），只能 `kill -9`。

## 6. 测量脚本（都能单独跑）

| 脚本 | 用途 |
|---|---|
| `tests/t_profile_req.py` | 复刻 `generate()` 逐段计时（定位固定开销） |
| `tests/t_prefix_cache.py` | 前缀缓存等价性（maxdiff 必须 0.000e+00） |
| `tests/t_prefill_split.py` | prefill 成本拆分（attention / MoE / 反量化地板） |
| `tests/t_batch_micro.py` | int8 GEMM 的批无关性微基准 |
| `tests/t_head_ab.py` | int8 head vs bf16 head（已证否） |
| `tests/t_ref_eval.py` | 参考 bf16 在同一文本上的逐 token NLL（配对比较） |
| `tests/t_chunk_state.py` | **分块 prefill 等价性**（非零状态算子级 + 端到端，要求 `0.000e+00`） |
| `tests/t_chunk_speed.py` | **分块 prefill 吞吐扫描**（含 MoE 置零对照，隔离 MoE 占比） |
| `tools/mcq_eval.py` | 24 题单选题任务级评测 |
| `tools/compare_ref.py` | 与参考 bf16 做逐 token 配对比较 |

## 7. 第 48 轮：分块 prefill 的边界（重要，避免重复踩坑）

分块 prefill 已实现，且与生产逐 token 路径**逐位等价**（`tests/t_chunk_state.py` 全部 `0.000e+00`，
含非零状态算子级与端到端分块=1/4/8）。但**单独上线是负收益**：

| chunk | 全量 ms/token | MoE 置零 | MoE 部分 |
|---:|---:|---:|---:|
| 1 | 213.60 | 155.96 | 57.6 |
| 8 | 93.16 | 33.02 | 60.1 |
| 32 | 78.37 | 18.11 | 60.3 |
| 128 | 75.83 | ~16 | ~60 |

- 稠密+注意力+DeltaNet 递推摊薄 **8.6×**（155.96 → 18.11，T=32 饱和）；
- **MoE 恒 ~60 ms/token，完全不摊薄**（仍是逐 token 调用、每 token 读自己那 10 个专家）；
- chunk=128 的 75.83 比生产（图捕获逐 token）**61.7 ms/token 慢 1.23×** —— 分块省下的稠密部分在生产路径上本来只有 ~1.7 ms/token。

⇒ prefill 与 decode 的瓶颈是同一件事：**MoE 专家权重读取**。上线顺序必须是
**MoE 并集 + 按专家分组 GEMM → 分块图捕获 → DeltaNet 32-token 递推**，预期 7–12 ms/token（60–100 tok/s）。
完成 MoE 分组前 `FNX_CHUNK` **保持 1**。

### 7.1 分组方案实测：拷贝不是瓶颈，**算子/调用个数才是**

真实 L00 专家权重、T=128、每 token 10 专家（并集 470/512），**独占 npu:0**：

| 路径 | ms / 128 token | ms/token/层 |
|---|---:|---:|
| A 逐 token（生产） | 102.3 | **0.80** |
| B 分组 + E-major 布局 | 377.9 | 2.95 |
| C 分组 + 现有布局 + `contiguous()` | 432.6 | 3.38 |
| 仅 `_wcat`（拷 10 专家 int4 权重） | — | 0.053 ms/次（占 A 的 13%） |

- 分组**反而慢 3.7×**：940 次调用 ≈ 0.40 ms/次，而单次算子只要 0.050–0.057 ms
  ⇒ 每专家约 0.35 ms 是簿记（gather/act/加权/`index_add_`）。
- 干净卡扫 `npu_weight_quant_batchmatmul`：**M 从 1 到 128、N=1280 时单次耗时完全不变**
  ⇒ 每次调用 ~50 µs 固定成本；因此**减少算子个数**才是主杠杆，减少权重流量是次要的。
- ⚠️ 口径：算子级测量必须在**独占卡 + `with torch.npu.device(...)`** 下做。
  在共享卡（npu:1）上量出的同一组数字被放大 1.8×，会得出"固定开销 0.6 ms"的假结论。

### 7.3 `FNX_MOE_FAST`：图内减算子零收益（默认 0）

按"算子个数是第一成本项"的推断实现了 4 处精简（`silu*mul` 全程 bf16、路由 6→3 算子、
`apply_diag` 对角一次 `index_select`、`_scat` 原地乘），每层少 8 个算子。
**实测 TPOT 63.7–63.9（对照 63.5–63.7）、`mean_nll` 2.1648（逐位相同）⇒ 零收益。**

原因：图捕获的 decode 路径里，被删的都是与 Cube GEMM **并行重叠**的 Vector 小算子，不在关键路径上。
⇒ **不要再用"减小算子"去优化 TPOT**；图内剩下的杠杆是 GEMM 结构（WQBM 15.8 ms/step = 31.7%）与内存/带宽。
开关 `FNX_MOE_FAST` 保留，默认 **0**。



### 7.2 运维两条（都是本轮踩出来的）

1. **报速度前先确认跑的是哪份启动脚本**：`pgrep -af fnx.server` 会打印完整环境。
   本轮 `p275.sh` 被改成过 `FNX_LQP=…-g128` + `FNX_DEQ_ATTN=10`，造成 TPOT 64 → 74 ms
   与 dev0 58.97 → 59.53 GiB（差值 0.55 GiB 正是 `DEQ_ATTN` 多占的量）的"假回归"。
2. **`systemctl stop` 不杀容器内的服务进程**（`docker exec -d` 不在 systemd cgroup 里），
   停服务后显存仍被占用；p275.sh 已加显式 `pkill -9 -f '[p]ython -m fnx.server'`。


