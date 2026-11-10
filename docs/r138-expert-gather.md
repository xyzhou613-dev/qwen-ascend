# Flash Next r138：专家权重收拢与布局转换融合

2026-11-10，192.168.18.206，2×Ascend 910B4-1、64 GiB/卡；CANN 9.1.0，驱动26.1.1，torch2.10.0 / torch_npu2.10.0.post4。

8k 正式部署 TPOT **39.16 ms**，相对本轮 r137 前测/回切基线提速分别为 **8.68% / 8.78%**。本轮优化不改变权重、量化方案或数学计算。

| 8064 输入 / 128 输出，各3轮中位数 | HTTP TTFT，s | TPOT，ms | 输出 token/s |
|---|---:|---:|---:|
| baseline | 4.328 | 42.56 | 23.496 |
| baseline-after | 4.330 | 42.60 | 23.474 |
| candidate8k | 4.329 | 39.06 | 25.602 |
| production8k | 4.327 | 39.16 | 25.536 |

| 262016 输入 / 128 输出，各1轮 | HTTP TTFT，s | TPOT，ms | 输出 token/s | 三事实召回 |
|---|---:|---:|---:|---:|
| baseline-after256 | 73.503 | 44.03 | 22.712 | 3/3 |
| candidate256 | 73.669 | 40.42 | 24.740 | 3/3 |
| production256 | 73.568 | 40.37 | 24.771 | 3/3 |

以上均为本机HTTP，全新文档，仅复用模板前缀。256k为单次验收，不是多轮统计；prefill路径未改，TTFT波动不应全部归因于本轮改动。r137到r138是增量对比，不是相对早期r119/初始版本的累计收益。

## 实现与适用范围

原 up 投影先分别 index_select 权重与scale，再分别 permute/transpose 并 contiguous。融合 kernel 依据动态 top-10 专家ID，直接把 expert-major 的原始W4打包位和BF16 scale搬到 K-major 连续输出；随后调用完全相同的CANN量化矩阵乘。

- 新增 `kernels/expert_gather/`、`fnx/expert_gather.py`；开关 `FNX_EXPERT_GATHER=1`，默认关闭。健康接口增加 `expert_gather`，版本 r138。
- 当前仅接入 decode gate/up：E=512、K=2560、N=1280、group=64、top10，expert-major权重与缓存BF16 scale。其他形状、布局与prefill自动保留原路径。
- 直接使用字节搬运，不解包W4、不重新计算scale、不改归约顺序；保留r137残差融合。27B配置与切换入口不变。
- 按逻辑数据量，去掉每token约1.67 GB的额外读写（48层的权重/scale中间转置）。这不是新增的HBM控制器实测数，不能直接从r136的14.61 GB中相减。

## 测试与计时修正

| 微基准 | 原路径 | 采用方案 | 校验 |
|---|---:|---:|---|
| Gather＋up量化矩阵乘 | 98.75 μs | 52.61 μs | 32组动态路由逐位一致 |
| 完整L00 decode | 约0.781 ms | 约0.693 ms | 输出、FP32状态、BF16历史逐位一致 |

真实L00还测试了128/512/2048行、各8组动态输入，全部逐位一致。三份8k完整请求在原版、候选与正式部署间逐字节一致；200-token NLL=2.165267。256k三事实与重复缓存通过。历史100题未在本轮重跑。

本轮重新验证了r128的直接搬运候选：历史单次graph replay约0.30ms，接近宿侧发射下限，不能据此否定设备内收益。新微基准每图16次运算，完整Gather＋up路径超过发射下限；孤立Gather的18.55μs仍接近该下限，应视作上界参考，不宣称精确kernel耗时。最终采用完整层和HTTP请求对照。

另试验了“UB内Gather重排＋连续写出”与流水重叠变体，Gather＋up约60.7μs，慢于52.6μs的直接搬运版本，未接入生产。代码与对照数据一同保存，避免重复试错。

## 部署、回退与证据

当前正式目录 `/root/ascend-dev/fnx/fnx`，端口9002，窗口262144；p275.sh持久启用FNX_EXPERT_GATHER=1和既有FNX_INJECT_NATIVE=1。模型切换管理器已恢复；临时禁止启动配置已移除。

```bash
cd /root/qwen-ascend
bash kernels/expert_gather/build.sh
# 将对应源码和fnx/libfnx_expert_gather.so部署到既有运行目录。
# 启动环境启用FNX_EXPERT_GATHER=1，通过既有modelctl入口重启。
curl -fs http://127.0.0.1:9002/health
```

健康检查应为service_revision=r138、expert_gather=true、inject_native=true、ctx=262144、worker_alive=true。只在当前软硬件矩阵验证；其他矩阵需要重建和验收。算子库登记为alpha，未做长期稳定性认证。

精确回退备份为 `/models/fnx-diagnostics/r138-expert-layout-20261110/deployment-backup/`；先停模型，再执行同目录 `rollback_files.py`，按原入口启动。也可将FNX_EXPERT_GATHER置0后重启，恢复r137专家路径。

远程证据目录 `/models/fnx-diagnostics/r138-expert-layout-20261110/`；本地附件 `r138-expert-layout-evidence/`。附件只收录白名单中的脚本、摘要与测试输出，不包含原进程私密环境文件、权重或图缓存。
