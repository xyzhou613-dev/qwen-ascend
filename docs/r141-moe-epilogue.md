# Flash Next r141：MoE 输出融合

验收完成：2026-11-11T09:14:32+08:00。实现提交a4eea951adaf53841c622b10bb64794422e926ff。

192.168.18.206，2×Ascend 910B4-1、64 GiB/卡；CANN9.1.0、驱动26.1.1、torch2.10.0 / torch_npu2.10.0.post4。262144窗口、9002端口。

8k正式 TPOT **34.75 ms**，较同轮r140前测/回切基线，输出速度提高 **6.22% / 6.30%**。

| 8064输入 / 128输出，各3轮中位数 | HTTP TTFT，s | TPOT，ms | 输出token/s |
|---|---:|---:|---:|
| baseline | 4.334 | 36.91 | 27.093 |
| baseline-after | 4.333 | 36.94 | 27.071 |
| candidate8k | 4.331 | 34.47 | 29.011 |
| production8k | 4.333 | 34.75 | 28.777 |

| 262016输入 / 128输出，各一次 | HTTP TTFT，s | TPOT，ms | 输出token/s | 事实召回 |
|---|---:|---:|---:|---:|---:|
| baseline256 | 73.095 | 38.11 | 26.240 | 3/3 |
| baseline-after256 | 73.637 | 38.25 | 26.144 | 3/3 |
| candidate256 | 73.584 | 35.73 | 27.988 | 3/3 |
| production256 | 73.653 | 35.99 | 27.785 | 3/3 |

HTTP端到端：8k三轮中位数 9.020 → 8.745 s；256k单次 78.496 → 78.224 s。

同机HTTP，全新文档，仅复用极短模板前缀。TTFT为客户端首token时间，TPOT为服务端统计。256k为单次验收，不代表多轮统计；实际262016输入+128输出，不是256k输入再加8k输出。prefill路径未改，TTFT差异不应直接归因于decode融合。

## 实现与数值边界

| 原路径 | r141 | 保留的舍入 |
|---|---|---|
| down Cartesian结果→对角张量 | 直接从10×10专家块读取10块对角 | 原矩阵乘及其BF16输出不变 |
| 路由权重转FP32→逐专家乘法→归约 | 同一设备内核完成 | 每专家乘积先BF16舍入，按原专家顺序FP32累加，结果BF16舍入 |
| 共享专家sigmoid/product→与归约相加 | 在同一内核中接续 | sigmoid BF16、乘积BF16、最终加法BF16 |

新增kernels/moe_epilogue与fnx/moe_epilogue.py。默认关闭，FNX_MOE_EPILOGUE=1启用；要求既有Cartesian down（FNX_DECODE_DN_GMM=0）、FNX_MOE_REDUCE=1和r140延迟共享门控。其他路径使用原实现。权重、路由、矩阵乘、RMS归约、KV精度和FP32递推状态均不变。

每层逻辑上省掉10×2560 BF16对角张量、10个FP32权重转换值和2560 BF16归约结果等中间张量。仅对角与归约中间值的写回/再读就约112640字节/层；这只是逻辑张量流量，不能当作实测HBM流量。收益主要针对调用与依赖开销，没有新的硬件计数器采样。

## 验证

128组动态组件输入逐位一致。真实L00 T=1/128/512/2048各8组动态输入，输出、FP32状态及BF16卷积历史逐位一致；三份8k完整输出跨前测、候选、回切与正式部署逐字节一致。
200-token NLL=2.165267；256k三事实召回和重复缓存通过。历史100题未重跑。

| 独占组件/层微基准 | 原路径 | 融合 |
|---|---:|---:|
| MoE输出组件 | 14.904 μs | 4.573 μs |
| 真实L00 decode | 0.64278 ms | 0.59571 ms |

组件每图64次操作，测量4次replay摊销；层测试为完整真实层。微基准不线性外推整模型收益。不宣称达到硬件极限。

另一个DeltaNet分块候选已保存到experiments/r141-delta-tiled：128步动态FP32状态/输出逐位通过，但初筛改善不足，64列切分更慢，未进入生产。初筛是r140常驻空闲时的组件测试，不作为正式性能结论。原生算子测试需使用当前stream，并保证异步输入准备已提交；首次缓存stream的无效测试已排除。

## 部署与回退

当前/root/ascend-dev/fnx/fnx；p275.sh仅新增本轮持久开关，其余现有参数保留。替换共享库前已停止其使用进程。模型切换管理器已恢复active，临时启动限制已移除。

本机编译器位于Flash Next容器，源码编译写入临时目录；其他机器使用实际容器名。

```bash
cd /root/qwen-ascend
FNX_R141_BUILD=$(docker exec bole-model-Flash mktemp -d /tmp/fnx-r141-build.XXXXXX)
docker cp kernels/moe_epilogue "bole-model-Flash:$FNX_R141_BUILD/moe_epilogue"
docker exec -e ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest bole-model-Flash bash "$FNX_R141_BUILD/moe_epilogue/build.sh" "$FNX_R141_BUILD/libfnx_moe_epilogue.so"
# 停模型、备份后安装配套engine/server/moe_epilogue.py和共享库。
python3 ops/r141-enable.py /root/ascend-dev/p275.sh --check
python3 ops/r141-enable.py /root/ascend-dev/p275.sh
# 通过既有modelctl入口重启Flash Next。
curl -fs http://127.0.0.1:9002/health
```

健康检查：service_revision=r141、moe_epilogue=true、post_native=true、expert_gather=true、inject_native=true、decode_down_gmm=false、ctx=262144、worker_alive=true。
停用本轮：FNX_MOE_EPILOGUE=0重启。完整回退：先停模型，执行/models/fnx-diagnostics/r141-moe-epilogue-20261111/rollback_files.py，再用原入口启动；精确备份在同目录deployment-backup/。

仅验证当前软硬件矩阵；其他910B环境需按已有自检确认矩阵、重编译并验收。原始证据/models/fnx-diagnostics/r141-moe-epilogue-20261111/；Git benchmarks/r141-moe-epilogue；本地附件r141-moe-epilogue-evidence/。附件仅取白名单，不含凭据、完整进程环境、权重或图缓存。
