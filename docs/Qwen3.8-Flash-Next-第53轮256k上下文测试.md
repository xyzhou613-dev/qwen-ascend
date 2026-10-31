# 第53轮：256k上下文验证

日期2026-10-31，192.168.18.206，2×910B4；基于已验收的r52稀疏QSA实现。
模型原生窗口262144，本轮不修改RoPE，不降低KV精度，也不减少模型层数。

## 容量调整

直接把128k配置的窗口改为262144、保持40+8层时，prefill图捕获失败：
GMM需申请1.58 GiB，设备0仅有1.23 GiB物理可用，已分配58.17 GiB。

改为38+10层，全部48层仍参与推理；图捕获通过。
设备0 allocated 54.873 GiB、reserved 57.426 GiB；设备1 allocated约15.39 GiB。
512-token prefill图、分组MoE、FP32 DeltaNet状态、BF16 KV和稀疏QSA均沿用r52。

## 验收口径

实际输入262016 tokens，给输出预留128 tokens；请求输出上限64 tokens。
新随机文档编号避免命中历史对话缓存，仅复用3-token静态模板前缀。
待检索事实分别位于正文约10%、50%、90%深度；要求三个项目的验收码均正确。
另测相同输入的缓存复用，要求新prefill为0且答案仍正确。

计时为服务已加载后的服务器本机HTTP请求，不含模型加载和外部网络传输。
三事实检索仅是本次输入的功能检查，不代表所有256k任务的准确率。

基础200-token质量检查：NLL=2.164792、PPL=8.7128、top1=48%，与r52一致。

## 完整结果

| 窗口 | 实际输入 | 新prefill tokens | 实际输出 | Prefill | 吞吐 | 首token | HTTP端到端 | 检索 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 131072（r52） | 130944 | 130940 | 44 | 289.236 s | 452.71 tok/s | 295.310 s | 299.961 s | 3/3 |
| 262144（本轮） | 262016 | 262012 | 44 | 607.678 s | 431.17 tok/s | 624.785 s | **630.217 s** | **3/3** |

256k完整输入已处理，记录出现`LONG_ACCEPT 262144`，无OOM或截断输入。
相同prompt再次请求，端到端7.074 s，新prefill为0，三个验收码再次全部正确。
本轮256k为单次新prompt实测，没有将缓存命中速度当成完整prefill速度。
相对128k，输入约翻倍，prefill约2.10倍、吞吐下降约4.8%。
端到端包含tokenization、前缀状态复制、长文状态保存、输出等开销，因此高于纯prefill时间。

## 原始记录

容量日志：`/root/ascend-dev/logs/r53_capacity_40.log`、`r53_capacity_38.log`。
长文日志：`/root/ascend-dev/logs/r53_long.log`，结果JSON：`/root/ascend-dev/r52/result_262144.json`。
控制脚本：`/root/ascend-dev/r53_256k.sh`；测试使用r52长文客户端的`TEST_CTX=262144`档位。
验证前128k配置备份：`/root/ascend-dev/r53/baseline.tgz`、`r53/baseline.env`。

## 最终运行配置

测试退出后恢复已验证的128k默认配置：`CTX=131072 LPC=40`。
`fnx.service`为active，`/health`确认ctx=131072、qsa_sparse=true；恢复后“3加5”回答8。
256k作为可选档安装至持久切换脚本，配置为`CTX=262144 LPC=38`。
切换会重启服务，应在没有进行中请求时执行：

```bash
bash /root/ascend-dev/r52_switch.sh 256k
# 恢复默认档
bash /root/ascend-dev/r52_switch.sh 128k
```

生产推理源码沿用r52，不需要为256k再修改模型算法。
保留128k默认，避免日常短请求承担256k更大的状态复制开销。
本地原始结果和日志位于`D:\Ascend learning\_work\r53-results`。
