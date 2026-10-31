# 第 55 轮：现网版本合入与独立验收

日期：2026-10-31
机器：192.168.18.206，2 × Ascend 910B4，48 层 Qwen3.8-Flash-Next（W4 MoE，约 121 G）

## 1. 本轮做了什么

现网从 **21:45 起被换成新引擎**（另一 AI 在 `baseline-r48.3` 之上改造：分块图 prefill、稀疏 QSA indexer、
expert-major grouped MoE、FP32 分块 DeltaNet、256k 上下文、每卡独立进程 24+24 流水），
`p275.sh` 于 22:08 同步替换，`fnx.service` 22:02 重启。**代码与文档当时只在服务器工作区，没有提交、没有推送。**

本轮的三件事：

1. 把该工作区（`fnx/` 4 个改过的模块 + `pipeline.py`/`pipeline_worker.py` 两个新模块 + 25 个测试脚本 +
   5 个脚本 + 6 份轮次文档 + `tools/repack_w4_em.py`）**提交进本仓库并打 tag `baseline-r55`**；
2. 对它做**独立验收**（不引用其文档的自测数字）：`/health`、`/v1/models`、真实请求 TTFT/TPOT、
   `/fnx/loss`（200 token）、24 题 MCQ；
3. 在技术文档新增 **§4.22**、README 状态段与本文件 §4，**修正 r48 两条被推翻的结论**，避免后人按旧结论放弃正确方向。

## 2. 独立验收结果（本仓库实测）

| 项目 | 实测 | 说明 |
|---|---|---|
| `/health` | `ctx=262144, pipeline_stages=2, worker_alive=true, prefill_chunk=4096, chunk_graph=true, qsa_indexer=true, qsa_sparse=true` | 结构符合 r54/55 文档 |
| `/v1/models` | `qwen3.8-flash-next-w4` | — |
| 显存 | dev0 **56.0 GB** / dev1 **56.6 GB** | 两卡均衡（旧形态 58.97 / 4.54 GiB） |
| 短请求（15 token prompt，48 输出） | TTFT **0.47 s**、**TPOT 54.1 ms**、`prefill_tok_s=28.3` | 短 prompt 的 prefill 有固定开销，不代表长文 |
| `/fnx/loss`（200 token） | **mean_nll 2.1653**、ppl 8.717、top1 48.5% | 旧基线 2.1648，差 +0.0005，远小于 0.01 门限 |
| MCQ 24 题 | **24/24，0.52 s/题** | 旧形态 24/24，3.48 s/题（分块图 prefill 带来的 6.7× 延迟改善） |

对照其自测（第 54–55 轮记录）：256k 新 prompt prefill **2,246.9 tok/s**、TTFT **117.86 s**、
decode **55.97 ms（17.87 tok/s）**、三事实检索 **3/3**；8k 新 prompt TTFT **6.99 s**、端到端 14.1 s。
我这边的独立实测与它同向、同量级（TPOT 54.1 vs 55.97 的差异来自 prompt 长度与统计窗口不同）。

## 3. 与本仓库 r48.3 的对比

| 指标 | r48.3 | r55 | 倍数 |
|---|---:|---:|---:|
| 上下文 | 2,048（硬上限 16k） | **262,144** | 128× |
| prefill（8k prompt） | ≈494 s（61.7 ms/token） | **6.99 s**（≈1,150 tok/s） | **71×** |
| prefill（256k prompt） | 不可用 | **117.86 s**（2,246.9 tok/s） | — |
| decode TPOT | 63.4–63.7 ms | **54.1 ms**（短请求） | 1.17× |
| NLL | 2.1648 | 2.1653 | 持平 |
| 长文检索 | 无法测（无 indexer） | **3/3 @256k** | — |

## 4. r48 两条结论的修正（重要）

1. **"分块 prefill 单独上线是负收益"** —— 只在**没有分块图捕获**时成立。r49 给 dev0 加固定块 NPUGraph
   （位置张量进图、块尾与缓存未对齐仍走原路径）后，分块 prefill 从 16.5 → 34.2 tok/s（2.07×），
   且 logits maxdiff=0、NLL 差=0。**结论应改为：分块必须与分块图一起上。**
2. **"按专家分组慢 3.7×，不应投入"** —— 那是 **per-expert 小调用 + WQBM** 的实测。
   r50 换成 **expert-major 权重布局 + `GroupedMatmulV5`**（WQBM 明确不支持三维 batch，会报
   `x dim should be [2,2]`）后，单层 T=128 从 44 ms → **7.5 ms**。**结论应改为：分组是正收益，
   前提是走 GMM 且把 CANN 的反量化工作区按专家 bucket 限流（512 专家曾申请 1.58 GiB）。**

另有两处历史判断一并修正：**16k 上下文上限**与 **dev1 不能图捕获**，都是**旧实现**的限制，
不是硬件限制（分别为稀疏 indexer + BF16 连续 KV、以及每卡独立进程所解决）。

## 5. 配置与回滚

- 启动：`scripts/prod_pipeline.sh`（= 现网 `/root/ascend-dev/p275.sh`，含显式杀容器内进程）；
- 环境详见技术文档 §4.22.3 / `BASELINE.md §8`；
- 新增权重目录：`/models/Qwen3.8-Flash-Next-LQP-w4a16-em`（由 `tools/repack_w4_em.py` 生成，纯转置、有逐位 round-trip 校验）；
- 回滚：`bash /root/ascend-dev/r55_rollback.sh`（回到 r53 路径，仍保留 256k）；
  部署前备份 `/root/ascend-dev/r55/predeploy.tgz`。

## 6. 已知边界（不要过度解读）

1. **256k 是单次新 prompt 的实测**，不代表所有输入形态、并发、或不同位置分布的延迟保证；
2. **decode 54–56 ms/token 是当前最大项**：expert-major + GMM 只用于 prefill，decode 仍逐 token 走 WQBM；
   我在 r48.3 上实测过 `FNX_MOE_FAST` 的"减算子"在图内**零收益**（§4.21.6）；
3. 长位置 decode 更慢（>2k 位置约 90 ms/token），索引器选块与 8k 窗口仍有成本；
4. `FNX_BATCH=1`，indexer 与分块均限单序列，**未做并发批量**——而"单次 WQBM 成本与 M 几乎无关"是实测事实，
   批量是把吞吐再推一档最直接的手段；
5. r50 起**全网已非逐位等价**（路由/累加顺序变化），质量靠配对 NLL/KL/top1 与长文检索门禁把关。
