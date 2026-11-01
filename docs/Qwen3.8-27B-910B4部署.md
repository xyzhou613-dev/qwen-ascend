# Qwen3.8-27B 部署到 2×910B4（vLLM-Ascend）

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
