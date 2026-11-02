# r107–109：逐位一致的原生FP32 DeltaNet decode

2026-11-02，在r106 / 390691a上继续。两卡910B4、48层24+24独立进程、262144上下文、8192主块、hybrid IPC；继承BF16 KV、W4专家、r95快照和r106逆排列排序。未启用MTP。

## 生产性能

服务器本机HTTP，模型已加载，全新随机文档；262016输入、128输出，新prefill262012，模板复用3 tokens。

| 次数 | 客户端TTFT | Decode TPOT | 输出速度 | HTTP端到端 |
|---|---:|---:|---:|---:|
| 1 | 92.450 s | 52.02 ms | 19.22 tok/s | 99.057 s |
| 2 | 91.960 s | 51.95 ms | 19.25 tok/s | 98.558 s |

r106平均TPOT 54.225 ms（18.442 tok/s）；当前51.985 ms（19.236 tok/s），按平均TPOT倒数比较，输出速度提高约4.31%。TTFT均值92.107→92.205秒；本轮主要收益是decode，不把接近波动的TTFT差异当作prefill加速。

新8064-token输入三次：TTFT中位数5.475秒，TPOT中位数50.53 ms。样本数量有限，非长期SLA或硬件极限证明。

额外持续生成测例：261120输入、预留1024输出，实际输出1024 tokens；TTFT 91.372秒、TPOT 51.99 ms（19.23 tok/s），三事实3/3。这个测例没有同形状的r106基线，不用于跨版本百分比计算。

## 精度与实现

r107用独立FP32张量对照CPU的多种累加顺序，确认当前CANN中间轴128行求和与逐行顺序相加完全一致。新AscendC内核在UB保留一头的FP32状态，执行原来的衰减、k点积、delta、外积更新、q点积；每次Mul/Add单独舍入，两个归约按原顺序进行。省去多个3 MiB中间张量和多次kernel调度，没有降低state精度或把归约换成近似BMM。

核心输入q/k/v/beta保持原BF16值，包装层只在NPU上精确转FP32；decay仍由原torch.exp计算。状态原址更新、地址不变，原snapshot/save/load/reset及图重放继续使用同一FP32张量。仅B=1、[1,48,128,128]连续FP32状态和原decode路径启用；大块prefill的DeltaNet算法未改。FNX_DN_NATIVE=0回到原路径，B>1或BMM分支保留原实现。

20/24/40核配置各256步递推，FP32状态、输出和有符号零按int32位型比较一致；真实L00的256次动态输入图重放，状态、卷积历史和BF16输出逐位一致。40核36次核心调用微基准约3.223→1.245 ms；同一个状态反复调用，不冒充36层完整模型延迟。早期默认dav-c220构建出现核配置相关异常；改为明确dav-c220-vec后各配置通过。生产只能使用Vector-only构建。

一次候选、两次生产新256k三事实均3/3；每轮重复n_prefill=0、追问仅新增29 tokens且回答7319；长文→重复→追问→短文→新长文，以及额外持续生成均通过，无新增OOM。四次200-token NLL均2.165267，参考2.164792，PPL8.7169、top1=48.5%；不把这些有限测例等同通用质量覆盖。

## 构建、部署、回退

源码kernels/delta_native/delta.cpp，构建bash kernels/delta_native/build.sh，在当前CANN环境生成fnx/libfnx_delta.so；不要用通用c220代替c220-vec。生成二进制未入Git，源码、构建命令、候选/重建二进制SHA256和生产验收记录均入库；生产使用实际通过验收的候选二进制。本轮跨目录重建库也通过20/24/40核、256步递推和有符号零检查。编译器会包含路径等元数据，不能仅凭跨目录二进制hash不同推断数学实现不同，后续重建应重新运行tests/r108_delta_probe.py和tests/r109_delta_layer.py。测试可用FNX_DN_NATIVE_SO指定待测库。

fnx.service active/enabled，health.delta_native=true，worker存活。bash /root/ascend-dev/r109_rollback.sh恢复r106；只关闭FNX_DN_NATIVE并重启也可回退算法。r95满窗口快照仍少存6.75 GiB；本轮未改变持久KV布局。
