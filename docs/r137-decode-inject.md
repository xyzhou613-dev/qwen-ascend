# Flash Next r137：decode 残差注入融合实测

2026-11-10，192.168.18.206；2×Ascend 910B4-1（64 GiB/卡）；CANN 9.1.0、驱动 26.1.1。代码提交 51c26e5。

## 结果

正式部署后，8k decode 输出速度较同轮回切原版基线提高 **2.02%**，TPOT 从 **43.36 ms 降到 42.50 ms**；相对前测基线提高 **1.62%**。未达到此前 20% 目标，也不代表硬件极限。

| 同机 HTTP，8064 输入 / 128 输出 | TTFT 中位数，s | TPOT 中位数，ms | 输出 token/s |
|---|---:|---:|---:|
| r135 原版，前测（各 3 轮） | 4.337 | 43.19 | 23.154 |
| 隔离候选（各 3 轮） | 4.344 | 42.78 | 23.375 |
| r135 回切复测（各 3 轮） | 4.342 | 43.36 | 23.063 |
| r137 正式部署（各 3 轮） | 4.332 | 42.50 | 23.529 |

| 262016 输入 / 128 输出 | HTTP TTFT，s | TPOT，ms | 输出 token/s | 三事实召回 |
|---|---:|---:|---:|---:|
| baseline-after256（各 1 轮） | 73.799 | 44.61 | 22.416 | 3/3 |
| candidate256（各 1 轮） | 73.638 | 44.07 | 22.691 | 3/3 |
| production256（各 1 轮） | 73.703 | 43.87 | 22.795 | 3/3 |

TTFT 表格来自独立请求，包含服务端预填与客户端流式观察开销。本次只改 decode 的残差注入；不能把几秒级的 prefill 波动解释为该算子的收益。256k 为每种配置单次验收，并非多轮统计结论。

## 改动与精度

- 将 `hyper + reshape(y * gate)` 融合为一个 AscendC kernel；先将乘积舍入为 BF16，再做 FP32 加法并输出 BF16，保留原链路的两次舍入。没有改成一次 FMA。
- 新增 `fnx/native_inject.py` 与 `kernels/inject/`；默认关闭，206 的 p275.sh 启用 `FNX_INJECT_NATIVE=1`。健康接口增加 `inject_native`，版本更新为 r137。
- 仅支持单 token、BF16、连续布局、hyper=[1,1,10240]、values=2560、gate=4。prefill、批量和其他类型/布局继续用原路径；未修改 27B 的模型配置或切换逻辑。
- 32 组动态微基准输入逐位一致；真实 L00 在 T=1/128/512/2048 各 8 组输入，输出、FP32 状态、BF16 卷积历史逐位一致。三份 8k 完整请求的输出在原版、候选和正式部署之间逐字节一致。
- 200 token NLL=2.165267；256k 三事实、重复请求缓存均通过。未重跑历史 100 题，也未宣称跨平台或长期稳定性认证。

## 为什么没有采用新的 BF16 GEMV

本轮实现了行式 Vector GEMV，但在重复权重和 512 MiB 权重轮换两种测试下都慢于 CANN MatMul，部分 BF16 输出位也因 FP32 归约顺序变化而不同，因此没有接入推理。原始测试代码和正反例数据保存在 `experiments/r137-gemv/`，避免后续重复踩坑。

微基准每张图包含 128 次运算，减少宿侧 graph replay 固定开销污染。第一版“一次 replay 只测一个小算子”的计时不能用来评估微秒级 kernel，已剔除。冷池轮换也不等于整模型性能，最终采用完整层和 HTTP 对照。

## 微基准与适用边界

| 输入 | 原链路 | 融合 kernel | 精度 |
|---|---:|---:|---|
| batch=1 | 7.282 μs | 2.525 μs | 32 组逐位一致 |
| batch=8（实验） | 7.804 μs | 5.024 μs | 32 组逐位一致；正式 wrapper 不启用 |
| batch=128（实验） | 16.473 μs | 47.497 μs | 逐位一致但更慢，故 prefill 回退 |
| 完整 L00 decode | 约 0.782 ms | 约 0.763 ms | 输出/状态/历史逐位一致 |

整模型提高约 1.6%～2.0%，不能将微基准的倍数或完整单层的差值直接乘层数当作请求收益。

## 部署与回退

正式路径为 `/root/ascend-dev/fnx/fnx`，端口 9002，ctx=262144；切换入口仍为 `/root/ascend-dev/modelctl.sh`。本轮补丁已写入 p275.sh，下次切换到 Flash Next 仍会启用。

同环境源码安装：

```bash
cd /root/qwen-ascend
bash kernels/inject/build.sh
# 将 engine.py、server.py、native_inject.py、libfnx_inject.so 部署到既有运行目录。
# 在既有启动参数中启用 FNX_INJECT_NATIVE=1，再通过 modelctl 重启 Flash Next。
curl -fs http://127.0.0.1:9002/health
```

健康检查应为 service_revision=r137、inject_native=true、ctx=262144、worker_alive=true。其他芯片/CANN 矩阵应重新构建并验收，不能直接套用此二进制。

206 的精确文件回退脚本：`/models/fnx-diagnostics/r137-gemv-20261110/rollback_files.py`，备份在同目录 `deployment-backup/`。回退时先停模型，执行文件回退脚本，再按原入口启动；或仅将启动参数置 `FNX_INJECT_NATIVE=0` 后重启，恢复原计算路径。

## 证据

- 远程：`/models/fnx-diagnostics/r137-gemv-20261110/`。本地：`r137-decode-inject-evidence/`。
- Git：`benchmarks/r137-decode-inject/` 保存微基准、真实层精度与请求摘要；`ops/registry.yaml` 收录算子，状态 alpha。
- 附件只收录显式白名单；未包含原进程环境文件、凭据、权重和庞大图缓存。
- 当前所有测试/采集进程已结束；正式模型和模型切换管理器保持运行。
