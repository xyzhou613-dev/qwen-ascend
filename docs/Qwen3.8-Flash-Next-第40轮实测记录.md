# Qwen3.8-Flash-Next 第 40 轮实测记录

> 两条线的收尾：
> **A. 便宜项** —— 把 shared-expert 与 gate 的 4 个权重再拼成一次 GEMM（数值不变）；另有两个"看起来该赚、实测不赚"的改动被回退。
> **B. MIX kernel** —— **多核正确性修好了**：`1024³` 正确且 24.9 TFLOPS，真实 MoE 形状单次调用 0.209 → **0.059 ms**。

---

## A. 生产（systemd 托管，TPOT 64.8–65.9 ms）

### A1 shared-expert + gate 合并（采纳，数值不变）

MoE 里 `sh_gate(640) / sh_up(640) / sh_gate2(1) / gate_w(512)` **四个权重都吃同一个 `x2`**，沿 N 拼成一次 GEMM（N=1793），层内切片：

```python
gu = lin(x2, self.sh_gu)                       # [b, 1793]
sh = lin(F.silu(gu[:, :MI]) * gu[:, MI:2*MI], self.sh_down)
sh = torch.sigmoid(gu[:, 2*MI:2*MI+1]) * sh
logits = gu[:, 2*MI+1:]                        # 路由 logits 也在这块里
```

- 少了 **2 次 GEMM 发射/层**（90 次/token）。
- `mean_nll` **2.1735 不变**（拼接前后同一组点积，只是 N 变了；这次恰好没动 nll）。
- TPOT：65.7–65.9 → **64.8–65.9**（跨重启的波动约 ±1 ms，收益在这量级内）。

### A2 两个"该赚没赚"的改动（已回退，留证据）

| 改动 | 预期 | 实测 | 处理 |
|---|---:|---|---|
| DeltaNet 去掉 `repeat_interleave`：把 `S` 看成 `[b,NKH,rep,HKD,HVD]` 直接广播（数值完全等价） | −0.9 ms | **+0.7 ms（更慢）** | 回退。5 维逐元素 (`sum(dim=-2)` + 双 unsqueeze 广播) 的效率不如"物化两次 + 4 维" |
| head 改 int8 分 4 段 | −3 ms | **+1.7 ms** | 回退（第 39 轮已记） |

> 教训：这台机器上 **"少一个 kernel"不等于"更快"**——张量维度/形状一变，Vector 单元的逐元素效率可能倒退。判断只认整请求 A/B。

### A3 r40 端到端

| 项 | 值 |
|---|---|
| TPOT | **64.8 / 64.9 / 65.7 / 65.9 ms（典型 65.5，15.3 tok/s）** |
| TTFT 热态 | 0.08 s |
| mean_nll / ppl / top1 | **2.1735 / 8.789 / 48.0%** |
| MCQ | **24/24** |
| dev0 显存 | 58.97 GiB |
| 累计 | 110.3（r35）→ 65.5 ms（r40）= **1.68×**；初版 318 ms 起算 **4.9×** |

> 对目标的诚实交代：原定"降到 62–63 ms"**没达到**，落在 65.5。原因是最有希望的那项（head int8，预期 −3 ms）实测反向 +1.7 ms，而剩下的合并项每个只有 0.5–1 ms 量级且已用尽。

---

## B. MIX kernel：多核正确性修好（本轮主要成果）

### B1 现象与定位

第 39 轮：单核数值 PASS、20 核 rel 0.95 FAIL，且 `y[0]` 与单核一致。

本轮把 **block 拓扑打印出来**（`PRINTF` 在 kernel 里），一眼看清：

```
[AIV Block 0/40] blk=0/20  usedCoreNum=20  singleM=205  singleN=16   (M=1024, N=64)
[AIV Block 1/40] blk=1/20  usedCoreNum=20  singleM=205  singleN=16
```

⇒ **tiling 把 M 和 N 都切了**（20 核 = 5 个 M 块 × 4 个 N 块，`singleN=16`），而 kernel 里每个核都把结果写到**偏移 0**，于是只有最后一个核的 tile 留下来，其余是过期/覆盖数据。

顺带排除了一个错误猜想：以为"真正执行 matmul 的是 AIC 侧"，于是在 kernel 里加 `#ifdef ASCEND_IS_AIV return;` → 结果**全零**（`y[0]=0.000000`，而"121 TFLOPS"是假的）。**AIV 侧的 KFC client 才是驱动方**，原结构是对的。

### B2 修法（kernel 自己加偏移 + SetTail）

```cpp
const uint32_t blk = GetBlockIdx();
if (blk >= coreNum) return;
const uint32_t nCoreNum = (N + sN - 1) / sN;      // 实测先铺 N 再进 M
const uint32_t m0 = (blk / nCoreNum) * sM;
const uint32_t n0 = (blk % nCoreNum) * sN;
gx.SetGlobalBuffer(x_gm + m0 * K, curM * K);
gw.SetGlobalBuffer(w_gm + n0,     K * N);
gy.SetGlobalBuffer(y_gm + m0 * N + n0, curM * N);
mm.SetOrgShape(M, N, K);
mm.SetTail(curM, curN, K);                        // 尾块
mm.SetTensorA(gx, false); mm.SetTensorB(gw, false); mm.IterateAll(gy, false);
```

### B3 结果（全部 PASS，rel ≈ 1.8e-4）

| 形状 | 修之前（多核，错） | **修之后（对）** | 峰值占比 |
|---|---:|---:|---:|
| 1024×64×64 | 0.044 ms FAIL | **0.024 ms PASS** | — |
| 1024³ | 0.111 ms FAIL | **0.086 ms → 24.9 TFLOPS** | 10.3% |
| **16×1280×2560**（真实 MoE gu 形状） | 0.209 ms FAIL | **0.059 ms → 1.8 TFLOPS** | — |
| 128×1280×2560 | 0.924 ms FAIL | **0.075 ms → 11.1 TFLOPS** | 4.6% |

对照基线：**我们现有 torch 路径在真实形状上是一层 0.043 ms（10 专家 bmm）**；自定义 op 现在这个"裸 matmul"是 0.059 ms——同量级。**真正的收益不在裸 matmul，而在把"解包 + gather + GEMM"融进同一个 kernel**：现在每 token 有 324 次 WQBM 调用（16 ms/步）、288 次 IndexSelect（5.3 ms/步），这些才是要消掉的对象。

### B4 下一跳（下一轮的活儿，已经很清楚）

1. **把 4-bit 解包写进 kernel**：输入 `q4`（v2/int4pack 容器）+ `s8/base`，在 kernel 内 `Vector` 侧解包到 L1/L0 的中间 buffer，再由 `Matmul` 吃 —— 这正是 ggml 的做法，也是唯一能把 WQBM 那 16 ms 消掉的路径。
2. **gather 也要进 kernel**：当前按专家 gather 权重（16.4 MB/层）在 kernel 外做；融进去才能省掉 IndexSelect 的 5.3 ms。
3. **单层 MoE gu 作为第一个真实目标**：输入 10 个专家 ids + 一行 x，输出 [10, 1280]，与现有 0.043 ms 基线比。

## C. 复现

```bash
# A（生产）
DW=1 ATTN=0 DH=1 SM=1 bash /root/ascend-dev/p331.sh     # 期望 64.8–65.9 ms / 2.1735 / MCQ 24/24

# B（MIX kernel，多核版）
docker exec fnx bash -lc 'cd /root/ascendc/cube_probe && bash build.sh && \
  cd build_out && bash custom_opp_ubuntu_aarch64.run --quiet'
docker exec fnx bash -lc 'cd /root/ascendc/cube_probe && \
  LT=/usr/local/Ascend/cann/aarch64-linux; V=/usr/local/Ascend/cann-9.1.0/opp/vendors/customize/op_api; \
  export LD_LIBRARY_PATH=$V/lib:$LT/lib64; \
  ./probe_host 1024 1024 1024 10; ./probe_host 16 1280 2560 10'
# 期望：两行都 PASS（rel 1.8e-4），1024³ ≈ 0.086 ms / 24.9 TFLOPS
```
