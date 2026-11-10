# Flash Next r139：decode down 投影优化

验收完成：2026-11-11T00:07:51+08:00；本轮目录按开始日期标记为20261110。代码2328330。

192.168.18.206，2×Ascend 910B4-1、64 GiB/卡；CANN9.1.0、驱动26.1.1、torch2.10.0 / torch_npu2.10.0.post4。

8k正式部署 TPOT **37.54 ms**；相对本轮r138前测与回切基线，输出速度分别提高 **3.97% / 4.21%**。

| 8064输入 / 128输出，各3轮中位数 | HTTP TTFT，s | TPOT，ms | 输出token/s |
|---|---:|---:|---:|
| baseline | 4.326 | 39.03 | 25.621 |
| baseline-after | 4.330 | 39.12 | 25.562 |
| candidate8k | 4.334 | 37.78 | 26.469 |
| production8k | 4.326 | 37.54 | 26.638 |

| 262016输入 / 128输出，各1轮 | HTTP TTFT，s | TPOT，ms | 输出token/s | 事实召回 |
|---|---:|---:|---:|---:|
| baseline-after256 | 73.782 | 40.40 | 24.752 | 3/3 |
| candidate256 | 73.818 | 39.13 | 25.556 | 3/3 |
| production256 | 73.691 | 39.03 | 25.621 | 3/3 |

同机HTTP、全新文档，仅复用模板前缀。256k是单次验收，不是多轮统计；prefill路径未改，TTFT波动不应归因于decode改动。本轮增量尚未达到20%，也没有宣称硬件极限。

## 实现

将既有专家原位搬运组件扩展到down的K=640/N=2560/G=10形状，直接输出K-major打包权重与scale。decode用同一批10行激活一次算出10×10组交叉输出，再取10组对角。保留原GMM路径，prefill仍走原分组计算。

- 生效：`FNX_EXPERT_GATHER=1`、`FNX_DECODE_DN_GMM=0`，并保留`FNX_INJECT_NATIVE=1`。共享库升级为支持gate/up与down两种形状的0.2版本。
- 中间逻辑乘加和输出数量增加，不能直接据此推断物理Cube计算量按相同比例增加或减少。小矩阵补齐、切分和数据布局都影响成本；本轮基于实测决定，不宣称已测得新的物理FLOPs。
- 动态专家ID顺序、W4打包位、BF16 scale值均保留。原头部实现、量化配置、KV精度及FP32递推状态未改。

## 精度与候选对照

| 测试 | 原路径 | 候选 | 结果 |
|---|---:|---:|---|
| 合成down完整路径 | 约102.2 μs | 约91.6 μs | 32组动态输入逐位一致 |
| 真实L00 decode | 约0.699 ms | 约0.650 ms | 输出、FP32状态、BF16历史逐位一致 |
| L00 prefill 128/512/2048行 | 原路径 | 原路径 | 每种8组动态输入逐位一致 |

三份8k完整请求在基线、候选、回切和正式部署之间逐字节一致；200-token NLL=2.165267；256k三处事实召回与重复缓存通过。历史100题未重跑。微基准每图16次操作，未用单次replay下限推算整模型收益。

未采用的候选也已保存：GMM的group_list_type、split_item、tuning_config扫描无稳定改善；融合expert-major down收拢单独省约6μs，但最终采用的对角路径更快。

真实lm_head试验中，BF16常驻约1.00ms，原W8约1.67ms；但少量logits末位不一致。对62080列真实切片用CANN单位矩阵还原的权重与CPU展开逐位一致，而输出仍有差异，定位到该切片的累加路径差异。因此保留原头部，没有部署这个更快但不逐位一致的候选。argmax在16组测试一致不等于通用准确率保证。

接口依据：本机torch_npu schema支持group_list_type、split_item与tuning_config。CANN中tuningConfig的预期每专家token数口径可参照[官方属性说明](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/850/API/aolapi/context/ops-transformer/aclnnGroupedMatmulWeightNz.md)；这些调参测试并未被选入生产。

## 部署、回退与证据

当前正式路径`/root/ascend-dev/fnx/fnx`、9002端口、262144窗口。p275.sh已持久写入选定参数；模型切换管理器恢复active，临时禁止启动配置已移除。替换.so前先停止其使用进程；未热覆盖已加载的共享库。

```bash
cd /root/qwen-ascend
bash kernels/expert_gather/build.sh
# 安装对应Python源码和共享库，保持FNX_EXPERT_GATHER=1，设置FNX_DECODE_DN_GMM=0。
# 通过既有modelctl入口重启Flash Next。
curl -fs http://127.0.0.1:9002/health
```

健康检查：service_revision=r139、expert_gather=true、decode_down_gmm=false、inject_native=true、ctx=262144、worker_alive=true。27B配置和模型切换入口不变。

回退down路径：将FNX_DECODE_DN_GMM置1后重启。完整文件回退：先停模型，执行`/models/fnx-diagnostics/r139-down-20261110/rollback_files.py`，再按原入口启动；精确备份位于该目录deployment-backup/。

Git证据位于benchmarks/r139-down/，未采用的候选位于experiments/r139-down/；远程原始目录/models/fnx-diagnostics/r139-down-20261110/，本地附件r139-down-evidence/。附件只取白名单，不含私密进程环境、凭据、权重和图缓存。
