# Qwen3.8-Flash-Next 第 39 轮实测记录

> 两条线并行：
> **A. 三个"便宜项"** —— 只减算子调用次数，不换算法。TPOT **67.7 → 65.8 ms**（15.2 tok/s），MCQ 24/24。
> **B. op-package MIX kernel 可行性** —— 给出决定性答案：**自定义 op 能跑到 Cube**（coreType=MIX、20 核 28.4 TFLOPS），流水线全通；剩多核数值待修。

---

## A. 便宜项（已上线）

### A1 三个候选，实测只有一个真正赚

| 改动 | 预期 | 实测 | 结论 |
|---|---:|---|---|
| head 改 int8 分 4 段（62080/段，算子上限 N≤65535） | −3 ms | **+1.7 ms（变慢）** | **弃用**：4 次小调用的固定开销吃掉省下的字节（0.64 GB vs 1.27 GB），单次大 bf16 GEMM 反而更高效 |
| full 层 q/k/v 合并成一次调用（`DenseW8.merge`） | −0.9 ms | 与下一项合并计 | 采纳，数值不变 |
| mixer 的 `down+inj`、shared-expert 的 `gate+up` 合并（bf16） | −1.8 ms | **−2.8 ms** | 采纳（+0.007 nats，见下） |

### A2 为什么合并有效

r38 之后重采 profile（设备 53.4 ms/步、**6,649 个 kernel/token、平均 8.0 µs**）：

| 算子 | 次/token | 平均时长 |
|---|---:|---:|
| `WeightQuantBatchmatmul` | 324 | 49 µs |
| `MatMul` | 531 | 13 µs |
| `Mul` | 1,204 | 6 µs |

`MatMul` 单次 13 µs 里绝大部分是固定开销，而 mixer 的 `w_down`（→320）与 `w_inj`（→4）吃的是同一个 `hn`、shared-expert 的 `sh_gate`/`sh_up` 吃的是同一个 `x2` —— 沿 N 拼成一次 GEMM 即可，层内切片对激活做（几十 KB）。

### A3 数值与取舍（诚实记账）

| 配置 | mean_nll | top1 | TPOT |
|---|---:|---:|---:|
| r38（不拼） | 2.1665 | 49.5% | 67.8–68.5 |
| 只拼 shared-expert | **2.1665** | **49.5%** | 67.3–67.6 |
| 只拼 mixer | 2.1735 | 48.0% | **65.4–65.7** |
| 都拼（**上线**） | 2.1735 | 48.0% | **65.7–65.9** |

- 数值变化的唯一来源是 **mixer 合并**：把 `[2560→320]` 变成 `[2560→324]` 的 GEMM，N 变了 →框架 tiling 变了 → 累加顺序变了。**不是量化变差**（权重逐位相同）。
- 判断：用掉的是 r38 攒下的"质量预算"（2.1665 是历史最好，比 r35 低 0.009），花 0.007 换 2.4 ms。**落点 2.1735 仍好于 r35 的 2.1756**；MCQ **24/24**（3.59 s/题，r38 是 3.69 s）。
- 开关：`FNX_SMALL_MERGE=1|0|mix|sh`、`FNX_DENSE_CAT=1|0`。想完全回到 r38 的 2.1665 只需 `FNX_SMALL_MERGE=0`（代价 +2 ms）。

### A4 r39 端到端（systemd 托管，多次）

| 项 | 值 |
|---|---|
| TPOT | **65.7 / 65.9 ms（15.2 tok/s）** |
| TTFT 热态 | 0.08 s |
| mean_nll / ppl / top1 | 2.1735 / 8.789 / 48.0% |
| MCQ | **24/24** |
| dev0 显存 | 58.97 GiB |
| 冷启动 | ≈30 s（装配 9.2 s） |

累计：**110.3（r35）→ 76.7（r37）→ 67.0（r38）→ 65.8 ms（r39）= 1.68×**。

---

## B. op-package MIX kernel 可行性（本轮重点）

### B0 背景：被卡住的到底是什么

第 26/27 轮的结论是"自研 AscendC 融合 kernel 只能跑 Vector（41.8 vs torch 26.8 ms），4 组 Cube 实验全部 507015"。那时走的是**直调 kernel**（自己 `bisheng -xcce` 编 + 自己 launch）。本轮改走 **op-package**（框架的官方流程），因为只有它能让框架接管 L1/L0 与流水。

### B1 流水线全通（这是本轮最重要的结论）

| 步骤 | 结果 |
|---|---|
| `msopgen gen -i cube_probe.json -f aclnn -c ai_core-Ascend910B4` | ✅ 生成工程 |
| 写 kernel：`Matmul<A_T,B_T,C_T>` + `REGIST_MATMUL_OBJ` + `IterateAll` | ✅ 编译通过（第一次报 "no matching member function for SetTensorA"，原因是传了裸 `__gm__` 指针，改成 `GlobalTensor` 即过） |
| 写 host：`MultiCoreMatmulTiling` + `SetBlockDim` + `InferShape` | ✅ 编译通过 |
| `bash build.sh` | ✅ 产出 `custom_opp_ubuntu_aarch64.run` + kernel `.o` + `libcust_opapi.so` |
| `custom_opp_ubuntu_aarch64.run --quiet` 安装 | ✅ SUCCESS |
| C++ 程序用 aclnn 调用（`aclnnCubeProbeGetWorkspaceSize` + `aclnnCubeProbe`） | ✅ rc=0，需链 `-lcust_opapi -lnnopbase -lacl_op_compiler -lascendcl` |
| 数值 vs CPU fp32 参考 | ✅ rel_mean **1.78e-04**（fp16 级） |
| kernel 元数据 `coreType` | ✅ **MIX**（Cube+Vector 混合编译，不是 Vector-only） |

> 也就是说：**"自定义 op 能不能跑 Cube" 的答案是能**，且是在框架正路上（aclnn + tiling 注册），不需要再跟直调死磕。

### B2 吞吐：核数与规模

| 形状 | 单核（`SetDim(1)`） | 20 核（`SetDim(20)`） | 说明 |
|---|---:|---:|---|
| 512³ | 2.2 TFLOPS | — | 太小，看不出 |
| 1024³ | 3.1 TFLOPS | **19.4 TFLOPS（8.0% 峰值）** | 20 核 → 6.3× |
| 2048³ | 3.2 TFLOPS | **28.4 TFLOPS（11.8% 峰值）** | 20 核 → 8.9× |
| 真实 MoE 形状 M=16,K=2560,N=1280 | 0.209 ms（≈0.5 TFLOPS） | 待做 | 这个形状是**发射/握手**主导，不是算力主导 |

对照：我们现有 torch 路径在真实形状上约 **1.5 TFLOPS**；`aclnnMatmul` 在 4096³ 能到 241 TFLOPS（基线报告）。所以 **28.4 TFLOPS 说明 Cube 真的在用**，只是离框架调优过的 kernel 还有 8 倍差距。

### B3 还没解决的（下一轮从这里接）

1. **多核数值错**：单核 rel 1.78e-4（PASS），20 核变 rel 0.95（FAIL），但 `y[0]` 与单核结果**完全一致** —— 即 0 号核的 tile 对，其余核写错位置。补 `mm.SetOrgShape(M,N,K)` 无效。下一步方向：per-core offset（`gm_c[offsetC]`）/ `SetWorkspace` / KFC 的 AIV↔AIC 映射。
2. **真实形状的单次调用成本 0.21 ms** 太高（我们分层调用 torch bmm 只用 0.043 ms）。要做成收益，必须**把一层的工作放进一个 kernel**（现在这个 op 只做了一次 bare matmul），否则每层一次 0.2 ms × 45 = 9 ms 反而更慢。
3. **解包要融进 kernel**：4-bit 的 q 与 scale 必须在 kernel 内参与点积（这正是 ggml 的做法），否则省下的只是发射，不是数据。

### B4 结论（投资判断）

- **技术可行性：已证明可行**（op-package + Cube + aclnn 全通，数值正确）。这解除了此前"工具链根本不让自定义 kernel 上 Cube"的怀疑。
- **性能天花板：还需两跳** —— ①修多核正确性 ②把"整层流式 + 解包融合"写进同一个 kernel。两跳都做完才有希望把 6,649 个 kernel 压到 ~200、逼近内存地板（≈3.7 ms/token）。
- **风险**：B3.2/B3.3 是数量级工作量（估计 1–2 周），且第二次跳（解包融合）本轮完全没验证。

---

## C. 未采纳/证否（本轮新增）

| 方向 | 数字 | 结论 |
|---|---|---|
| head 改 int8 分段 | 65.4 → 67.1 ms（慢 1.7） | 弃用 |
| `FNX_DEQ_ATTN=20`（再多预反量化 20 层） | 启动期驱动 OOM | 弃用：`hbm0` 显示的 2 GiB 余量不是真可用 |
| LPC=46（挪一层进图） | 要 802 MiB，只 708 MiB | 弃用（第 38 轮已记） |

## D. 复现

```bash
# A 组（生产口径）
DW=1 ATTN=0 DH=1 SM=1 bash /root/ascend-dev/p331.sh     # 期望 65.7–65.9 ms / 2.1735
# 回到 r38 的质量点：SM=0 → 67.3–67.6 ms / 2.1665

# B 组（MIX kernel）
docker exec fnx bash -lc 'cd /root/ascendc/cube_probe && bash build.sh && \
  cd build_out && bash custom_opp_ubuntu_aarch64.run --quiet'
docker exec fnx bash -lc 'cd /root/ascendc/cube_probe && \
  LT=/usr/local/Ascend/cann/aarch64-linux; V=/usr/local/Ascend/cann-9.1.0/opp/vendors/customize/op_api; \
  export LD_LIBRARY_PATH=$V/lib:$LT/lib64; ./probe_host 2048 2048 2048 10'
# 期望：coreType=MIX、单核 PASS、20 核 28.4 TFLOPS（多核数值仍 FAIL，见 B3.1）
```
