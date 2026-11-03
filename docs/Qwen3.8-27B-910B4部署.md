# Qwen3.8-27B 部署到 2×910B4（vLLM-Ascend）

> 2026-11-03 更新：`modelctl.sh start 27b` 可使用 BF16 TP=2 离线分片。见[离线分片与切换实测](#27b-lqp)。

日期：2026-11-01
机器：192.168.18.206，2 × Ascend 910B4（每卡 64 GB），容器 `fnx`（镜像 `quay.io/ascend/vllm-ascend:v0.23.0`）

## 1. 现状（已部署并通过冒烟）

| 项 | 值 |
|---|---|
| 服务 | vLLM-Ascend **0.23.0**，`vllm serve`，端口 **8000** |
| 模型 id | `qwen3.8-27b` |
| 权重 | `/models/Qwen3.8-27B`（bf16，18 分片，**52 GB**） |
| 并行 | `--tensor-parallel-size 2`，每卡权重 25.2 GB，进程占用 **57.2 GB/卡** |
| 上下文 | `--max-model-len 32768`（模型本身支持 262144，可加） |
| 实测 | 稳态 **TPOT 37.2 ms ≈ 26.9 tok/s**（首请求含图编译，约 50 s） |
| 多模态 | 视觉塔已加载；启动时用 `--limit-mm-per-prompt '{"image":0,"video":0}'` 只走纯文本 |

冒烟：

```bash
curl -s http://127.0.0.1:8000/v1/models | head -c 200
# {"id":"qwen3.8-27b", ... "max_model_len":32768 ...}
```

## 2. 为什么用 bf16 而不是 w8a8

201 上另有一份 `Qwen3.8-27B-w8a8`（29 GB），但它是 **MindIE / msModelSlim 格式**
（`quant_model_description.json` 里是 `W8A8_DYNAMIC` + `*.weight_offset` 条目，`group_size: 0`），
**vLLM-Ascend 没有对应 loader**，直接加载会失败。因此在 910 上用标准 HF 的 **bf16** 版本。

## 3. 权重怎么来的

201（RTX PRO 6000）→ 910 直传（同网段可达，910 装了 `scp` 但没有 `rsync`，所以用 `tar | ssh`）：

```bash
# 201 上（已配置免密：~/.ssh/id_910 → 910 的 /root/.ssh/authorized_keys）
tar cf - -C /home/server/bole Qwen3.8-27B \
  | ssh -i ~/.ssh/id_910 root@192.168.18.206 "mkdir -p /models && tar xf - -C /models"
```

实测 **约 110 MB/s**，52 GB 用时约 8 分钟；两岸分片总字节一致（55,563,006,776 B）。

## 4. 启停

启动脚本 `/root/ascend-dev/start_vllm_27b.sh`（容器内运行）：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 vllm serve /models/Qwen3.8-27B \
  --served-model-name qwen3.8-27b --tensor-parallel-size 2 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --host 0.0.0.0 --port 8000 --trust-remote-code \
  --limit-mm-per-prompt '{"image": 0, "video": 0}'
```

```bash
# 起
docker exec -d fnx bash -lc "bash /root/ascend-dev/start_vllm_27b.sh"
# 日志
tail -f /root/ascend-dev/logs/vllm27b.log
# 停
docker exec fnx bash -lc "pkill -f 'vllm serve'"
```

**注意**：`max_model_len` 与请求的 `max_tokens` 共享预算。32768 上下文下，
`max_tokens=32768` 的请求会被 400 拒绝（prompt 无处安放）；评测时把输出上限设为 16384 即可，
或重启时把 `--max-model-len` 调大。

## 5. 思考链默认是开的

默认请求会输出可见推理（`用户要求…`）。要关掉（与 Flash-Next 的 `FNX_ENABLE_THINKING=0` 对齐）：

```json
{"chat_template_kwargs": {"enable_thinking": false}}
```

实测关闭后回答更短（同题 30 tok vs 64 tok）。

## 6. 与 Flash-Next 的关系

**两者不能同时常驻**：Flash-Next（r55）占两卡各 ~54–57 GB，27B bf16 也占 57 GB，122 GB 装不下两份。
切回 Flash-Next：

```bash
docker exec fnx bash -lc "pkill -f 'vllm serve'"; sleep 5
systemctl start fnx.service          # 会跑 /root/ascend-dev/p275.sh
curl -s http://127.0.0.1:8000/health
```

切走时请务必先杀 vLLM 进程：**`systemctl stop` 不会杀容器内的进程**（见技术文档 §7.2），显存不释放。

## 7. 为什么这条路可行（证据）

| 检查 | 结果 |
|---|---|
| vLLM 0.23 注册表 | `"Qwen3_5ForConditionalGeneration": ("qwen3_5", "Qwen3_5ForConditionalGeneration")` |
| linear attention | `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`（`QwenGatedDeltaNetAttention`） |
| Ascend 算子 | vllm-ascend 自带 `aclnn_chunk_gated_delta_rule_fwd_h`、`aclnn_fused_gdn_gating`、TBE `recurrent_gated_delta_rule`（ascend910b） |
| 官方 e2e | vllm-ascend CI 有 `test_qwen3_5.py`、`test_qwen3_5_0_8b.py`（单卡/四卡） |

模型本身（读 `config.json`）：64 层、hidden 5120、24 头 / 4 KV、head_dim 256、dense FFN 17408、
词表 248,320、每 4 层 1 个 full attention（其余 linear/DeltaNet）、带视觉塔；dense 27B，非 MoE。

<a id="27b-lqp"></a>
## 8. BF16 离线分片与模型切换

27B 采用 vLLM 的 `sharded_state` 格式。导出时用原始 loader 完成 TP 切分和权重合并，在 Ascend 把权重转成算子布局之前按 rank 保存。启动时每张卡只读取自己的分片，仍走同一套 Ascend 权重后处理。模型保持 BF16、TP=2 和 32768 上下文，生成接口及模型 ID 与原部署相同。[vLLM Sharded State](https://docs.vllm.ai/en/stable/api/vllm/model_executor/model_loader/sharded_state_loader/)

快速配置单独缩减启动图准备，启用 `FULL_DECODE_ONLY` 并捕获 1/2/4/8 四种大小。Prefill 和超过捕获范围的批次走 eager。调度并发上限仍由 vLLM 配置控制。[vLLM-Ascend 0.23 图模式](https://docs.vllm.ai/projects/ascend/en/v0.23.0/user_guide/feature_guide/graph_mode.html)

```bash
cd /root/ascend-dev
./modelctl.sh start 27b
./modelctl.sh start flash-next

# 保留离线权重，恢复原图配置
MODELCTL_DISABLE_LQP_FAST=1 ./modelctl.sh start 27b
# 恢复原始权重与原图配置
MODELCTL_DISABLE_LQP=1 ./modelctl.sh start 27b
```

`models.json` 的 `lqp_path` 指向 `/models/Qwen3.8-27B-LQP-tp2`。启动前在容器内检查源权重、模型元数据、双 rank 产物和软件版本。产物缺失或校验失败时，脚本走原始权重与原启动参数。

离线导出需要独占双卡。将 `scripts/lqp_27b.py` 和 `scripts/lqp_27b_worker.py` 放在容器可见的 `/root/ascend-dev`，使用同一运行环境执行。新产物先写入临时目录，完整校验后再发布到目标路径。已有产物时，用新目标目录导出，确认后更新配置。

```bash
./modelctl.sh stop
docker exec fnx bash -lc '
  export ASCEND_RT_VISIBLE_DEVICES=0,1 VLLM_WORKER_MULTIPROC_METHOD=spawn
  export PYTHONPATH=/root/ascend-dev:${PYTHONPATH:-}
  python3 /root/ascend-dev/lqp_27b.py build \
    /models/Qwen3.8-27B /models/Qwen3.8-27B-LQP-tp2
'
./modelctl.sh start 27b
```

测量脚本从执行切换命令计时，轮询目标模型 ID，并发起本机流式请求。首个非空生成片段计作切换完成。

```bash
python3 bench_model_switch.py 27b /tmp/flash-to-27b.json
python3 bench_model_switch.py flash-next /tmp/27b-to-flash.json
```

### 2026-11-03 实测

双 910B4，27B 为 BF16 TP=2 文本模式，保留系统正常页缓存。原版 Flash Next → 27B 从切换命令到脚本健康检查为约 **238 秒**，首次流式请求 TTFT 为 **0.52 秒**。优化后两轮到目标模型接口就绪为 **179.23 / 181.48 秒**。

| 轮次 | Flash Next → 27B 首 token | 27B → Flash Next 首 token | 两方向合计 |
|---|---:|---:|---:|
| 1 | 179.86 s | 188.98 s | 368.85 s |
| 2 | 182.14 s | 189.74 s | 371.89 s |

切回 Flash Next 使用 256k 配置，两轮的启动图捕获分别为 153.7 / 154.8 秒。两方向合计只累加两次切换，轮中精度和解码探针单独计时。原始记录在服务器 `/root/ascend-dev/lqp27/`。

| 27B 阶段 | 原版 | 优化第 1 轮 | 优化第 2 轮 |
|---|---:|---:|---:|
| 权重加载 | 7.76 s | 2.09 s | 2.04 s |
| 图捕获 | 51 s | 5 s | 6 s |
| 单请求 64-token 解码中位数 | 27.65 tok/s | 27.71 tok/s | 27.70 tok/s |

三条固定提示的贪心生成文本和 67 个返回 token 一致。所选 token 的 logprob 最大绝对差为 0.0000122。图配置保留 `mode=3` 以初始化当前 Ascend 版本的 ACLGraph 工作区。只将 `mode` 改为 0 并保留全解码图，在本机曾因未初始化工作区而失败。
