# 第50轮：面向8k prompt的批量prefill

日期：2026-10-31。服务器192.168.18.206，2×Ascend 910B4。

## 本轮起点：最终r49真实长文基线

请求7830 token（其中7826个新prefill token），prefill315.328 s，24.82 tok/s；
TTFT317.005 s，总耗时317.544 s。末尾3+5问题回答正确。
重复请求总耗时0.654 s（客户端0.689 s），没有重新prefill。
NLL2.164792、PPL8.7128、top1=48%，短位置decode65.97 ms/token。
日志：`/root/ascend-dev/logs/r50_baseline.log`。

## 已定位的热点和接口约束

1. indexer的`bool.repeat_interleave(4)`：T=64时约81 ms。
   改为`unsqueeze/expand/reshape`后约0.334 ms，掩码逐位一致。
   位置0、3、2047、2048、2051、4095、8191及动态图重放均通过。
2. 原MoE逐token，单层T=128约44 ms；真实权重grouped matmul约7.5 ms。
   T=512约174 ms对11.1 ms。此处是单层隔离测试，不能当作全网吞吐。
3. WQBM不支持三维batch输入，本机明确报`x dim should be [2,2]`。
   GroupedMatmulV5的stack输入则支持int4及三维group scale；list输入需要额外offset。
4. CANN grouped matmul会申请较大的临时反量化空间，完整512专家的down投影曾申请1.58 GiB。
   必须限制每次处理的专家数量，并计入驱动工作区，而不能只看torch allocated。
5. 分批输入切片曾使真实MoE误差从0.12%增至约53%。将输入与权重复制到storage offset=0的
   有界缓冲后，分批/整批/去空组结果恢复一致的约0.12%舍入级差异。
6. int64 ArgSort会退到AI CPU；专家ID仅0..511，转fp32后排序不损失整数信息。

## 候选实现

- 新增expert-major权重布局，原artifact不删除。转换为纯转置，每层都有整数逐位round-trip检查。
  新目录：`/models/Qwen3.8-Flash-Next-LQP-w4a16-em`，48层。
- MoE整块路由，按专家排序，两个grouped matmul，逆排列后按原top10顺序加权。
- 分组临时缓冲按专家bucket限制，路由边界一次复制到host供两次投影复用。
  这一模式使用eager prefill调度；decode保留NPUGraph。
- DeltaNet使用FP32分块公式，保留FP32递归状态。
  早期全局矩阵乘积版本在随机key下T=64相对状态误差2.6e-7、约0.737 ms，
  但后续真实相关key测试暴露不稳定性，已淘汰。普通三角求解约10 ms。
  最终稳定化算法见下节；大块内部以64-token子块推进，限制工作区。
- QSA按GQA的两个KV head组织BMM，避免复制到24个head；query按64个位置分块，控制8k注意力临时矩阵。

原生`npu_chunk_gated_delta_rule`仅支持BF16状态，本轮未将它作为默认候选；
它的算子局部误差约0.3%–0.4%，比FP32分块路径大。

## 正确性与上线原则

候选需与同一权重的原单token路径对照。检查继续teacher-forcing的NLL与top1，
覆盖短位置、跨2048边界及实际长请求。不能用局部算子速度或HTTP 200判定成功。
前面的错误分批版本已被质量门禁拒绝，没有上线。

## 回滚与资产

本轮起点备份：`/root/ascend-dev/r50/baseline.tgz`，环境`baseline.env`。
隔离开发包：`/root/ascend-dev/r50/fnx`。
线上源包：`/root/ascend-dev/fnx/fnx`；仓库：`/root/qwen-ascend`。
所有测试日志：`/root/ascend-dev/logs/r50_*.log`。

## 进一步稳定化及整块图

完整64行有限级数在随机key下精确，但真实相关key会形成很大的中间幂，已拒绝这一版本。
最终候选使用8×8对角块预处理，再求解最多8级的块下三角系统；不再直接求全64行的幂。
相关key、beta=1、微小衰减压力测试：T=64/128/512全部有限，输出相对差≤2.2e-6，状态相对差≤7.4e-6。

为容纳CANN GMM工作区，实验确认LPC=43仍不足，LPC=42通过整块图捕获。
总层数仍为48；dev0为42层，dev1为6层及head。

全网512-token整块图测试（P=8192）：566.32 tok/s。
此数为短prompt的全网prefill吞吐，不能直接当作真实8k结果。
继续teacher-forcing128 token：top1全部一致，NLL差+0.000515、KL=0.001422。
从位置2040开始再跑512-token块并继续128 token：top1全部一致，NLL差+0.001625、KL=0.000659。
全网logits存在舍入/路由差异，不是逐位等价；本轮以配对NLL、KL、top1及实际检索验收。

候选生产配置：

```bash
FNX_LPC=42 FNX_NLAYERS=48 FNX_CTX=8192
FNX_CHUNK=512 FNX_CHUNK_GRAPH=1 FNX_INDEXER=1
FNX_MOE_GROUP=1 FNX_MOE_GROUP_SPLIT=0 FNX_QSA_GQA=1
FNX_PREFILL_DN=fp32_chunk FNX_DELTA_SOLVER=blockmatmul
```

## 最终线上验收（已部署）

与本轮起点同一条7830-token请求、新启动服务、只有3个静态模板token复用：

| 指标 | r49基线 | r50最终 |
|---|---:|---:|
| 新prefill token数 | 7826 | 7826 |
| 实际prefill时间 | 315.328 s | **14.009 s** |
| 实际prefill吞吐 | 24.82 tok/s | **558.63 tok/s** |
| TTFT | 317.005 s | **14.235 s** |
| 总耗时（7个输出token） | 317.544 s | 14.789 s |
| 同prompt缓存复用总耗时 | 0.654 s | 0.665 s |
| 短请求decode TPOT | 65.97 ms | **89.32 ms** |

真实长文prefill提速约**22.5倍**。短decode变慢是为CANN工作区把3层从dev0移到dev1的代价，
必须与prefill收益一起报告。长位置该7-token请求TPOT为92.23 ms。

质量基准：NLL=2.164792、PPL=8.7128、top1=48%（200 token），与基线相同。
完整长文回答末尾的3+5问题正确，缓存复用回答一致，越界请求返回400。
另一个7480-token新prompt，在约10%/50%/90%深度放入三个项目验收码：**3/3取回正确**；
该次prefill13.318 s、561.36 tok/s。此项是一次三事实检索，不是完整长上下文能力评测。

健康接口ctx=8192、prefill_chunk=512、chunk_graph=true、qsa_indexer=true；
systemd active/enabled。运行源码与离线重排工具已同步到本地工作区及远端仓库工作区，未推送。

原始证据：`r50_baseline.log`、`r50_model_graph42.log`、`r50_online.log`、`r50_needle.log`。
代码MD5：

```text
engine.py 3b2f451996ccb130d7574ce1baa4cf18
model.py  828b41d0ffbb40161048c68678eb8981
server.py d19d67f26b60f24698983666bd1cb862
load.py   05d62485cd8a34fc27c58b6c721c5e14
```

## 与Strata的对照和剩余差距

Strata v0.1.41社区报告：Ryzen AI Max+395/Radeon8060S、128GB统一内存、70W、
UD-IQ4_XS、8192个新token，6次中位数969 tok/s（约8.5秒）。
来源：https://github.com/Niko1221/Strata/blob/main/bench/results/2026-10-09-community-strix-halo-laptop-70w/README.md

本轮实测558.63 tok/s，约为该报告的58%，吞吐差约1.7倍。
硬件、量化、输入、PLE策略及测量口径不同，不能视为同平台公平排名。
后续主要空间在GMM工作区/反量化访存、FP32 DeltaNet块求解融合，以及dev1 eager decode。
