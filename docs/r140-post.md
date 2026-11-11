# Flash Next r140：decode 门控与后处理融合

验收完成：2026-11-11T08:28:38+08:00。实现提交8afe37c、94b8c44。

192.168.18.206；2×Ascend 910B4-1、64 GiB/卡；CANN9.1.0、驱动26.1.1、torch2.10.0 / torch_npu2.10.0.post4。262144窗口，9002端口。

8k正式部署 TPOT **36.88 ms**，相对同轮r139前测/回切基线，输出速度分别提高 **2.03% / 2.17%**。

| 8064输入 / 128输出，各3轮中位数 | HTTP TTFT，s | TPOT，ms | 输出token/s |
|---|---:|---:|---:|
| baseline | 4.326 | 37.63 | 26.575 |
| baseline-after | 4.333 | 37.68 | 26.539 |
| candidate8k | 4.332 | 36.96 | 27.056 |
| production8k | 4.332 | 36.88 | 27.115 |

| 262016输入 / 128输出，各1轮 | HTTP TTFT，s | TPOT，ms | 输出token/s | 事实召回 |
|---|---:|---:|---:|---:|---:|
| baseline-after256 | 73.734 | 39.05 | 25.608 | 3/3 |
| candidate256 | 73.683 | 38.29 | 26.116 | 3/3 |
| production256 | 73.717 | 38.12 | 26.233 | 3/3 |

TTFT来自同机HTTP客户端；TPOT来自服务端统计。全新文档，仅复用极短模板前缀。256k为单次验收，不是多轮统计；实际输入262016、输出128，不代表256k输入加8k输出。decode融合没有带来显著TTFT改善。

## 改动与精度边界

| 融合组件 | 保留的数值边界 |
|---|---|
| HC activation / injection gate | 除4后BF16舍入、SiLU的Div路径、Sigmoid后的BF16舍入与乘2 |
| Sigmoid + 四分支weighted mean | 权重BF16、乘积BF16、原有四分支FP32求和顺序 |
| post-RMS乘gamma与转换 | 原RMS reduction和epsilon不变；DeltaNet先保留归一化输出BF16舍入，再乘门控 |
| 共享专家门控与主专家相加 | Sigmoid BF16、乘积BF16、最后加法BF16 |

仅已验证的单token形状进入新路径；其他输入回退原实现。prefill分块路径保留；即时释放门控logits，避免延长大输入临时张量生命周期。没有更改权重、量化、KV格式或FP32递推状态。

覆盖全部65280种有限BF16位型在对应SiLU/门控角色中的输入，并通过32组动态组件校验；两种post-multiply分别64组。该覆盖不等于全部多输入笛卡尔组合。真实L00在T=1/128/512/2048各8组动态输入中，输出、FP32状态与BF16卷积历史逐位一致。

三份8k完整输出在前测、候选、回切、正式部署之间逐字节一致。200-token NLL=2.165267；256k三事实召回、重复缓存通过。历史100题未重跑；未做新的硬件计数器profiling，不能据本轮吞吐声称HBM或计算单元已跑满。

## 单组件实测

| 组件 | 原路径，μs | 融合，μs |
|---|---:|---:|
| HC激活和注入gate | 5.927 | 1.157 |
| Sigmoid + 四分支归约 | 3.228 | 1.929 |
| 共享专家门控合并 | 2.859 | 1.603 |
| RMS后乘gamma/转换 | 2.397 | 2.209 |
| 保留BF16边界的DeltaNet门控 | 3.977 | 1.683 |

真实L00 decode：0.65536 → 0.63436 ms。微基准按图内256/512次操作摊销；不能线性外推整模型收益。单独RMS后Mul+Cast接近持平，主要收益来自调用融合。

未采用激活的reciprocal与重结合变体：BF16末位出现差异；生产固定mode0 Div。仍未达到单轮20%目标，也不宣称硬件极限。

## 部署、回退与复用

运行目录/root/ascend-dev/fnx/fnx。新增post_ops.py、libfnx_post.so，engine/server需配套升级。FNX_POST_NATIVE默认关闭，本机p275.sh以参数化默认1持久启用；保留现有环境覆盖及其他启动参数。前置参数化/旧式脚本适配检查和bash语法检查均通过。共享库在模型停止后替换。

```bash
cd /root/qwen-ascend
bash kernels/post_ops/build.sh
# 停止模型后安装配套源码与共享库；按增量升级说明备份、校验。
python3 ops/r140-enable.py /root/ascend-dev/p275.sh --check
python3 ops/r140-enable.py /root/ascend-dev/p275.sh
# 通过既有modelctl入口重启Flash Next。
curl -fs http://127.0.0.1:9002/health
```

健康检查：service_revision=r140、post_native=true、expert_gather=true、inject_native=true、decode_down_gmm=false、ctx=262144、worker_alive=true。切换管理器已恢复active，临时启动限制已移除。

仅回退本轮后处理：FNX_POST_NATIVE=0重启。完整文件回退：先停Flash Next，执行/models/fnx-diagnostics/r140-hc-20261111/rollback_files.py，再按原入口启动。精确备份位于同目录deployment-backup/；首次脚本适配失败的回退记录另存attempt1，未覆盖备份。

仅验证本机软件矩阵与形状；其他910B环境先按部署自检确认矩阵，重新编译并运行组件/真实层和完整请求验收。

Git：benchmarks/r140-post/，experiments/r140-post/。远程原始记录/models/fnx-diagnostics/r140-hc-20261111/；本地白名单附件r140-post-evidence/，不含凭据、进程环境、权重或图缓存。
