# 第52轮：32k、64k、128k上下文

日期2026-10-31，192.168.18.206，2×910B4。
起点为r50生产配置：8192窗口、512-token prefill图、42+6层。
模型原始`max_position_embeddings=262144`，没有扩展RoPE或修改模型原生窗口。

## 原执行路径的容量失败

- 32768：图捕获时申请1.58 GiB的GMM工作区失败，物理可用约1.30 GiB。
- 65536：运行时设备内存申请失败，异步报在ConstantPadNd。
- 131072：decode注意力把两头KV复制到24头时，申请1.50 GiB失败。

因此不是只改`FNX_CTX`就能支持。

## 优化

新增`FNX_QSA_SPARSE=1`：indexer直接返回最多2048个选中token，加至多3个尾token。
位置排序后直接gather对应KV，GQA仅计算这批token；避免完整窗口的多头KV复制和稠密注意力矩阵。
索引器按32个query分块打分，限制128k窗口下的临时空间。
单token KV更新改为对应行写入，避免每步广播重写完整KV。

缓存快照新增直接复制到host的路径，避免先在NPU克隆一份完整KV。
128k的12层BF16 KV合计3 GiB，另有indexer key和递归状态；这项改动避免额外显存峰值。
KV仍为BF16，没有通过降低KV精度腾空间。

## 正确性和启动门禁

在8192、32768、65536、131072窗口的2040附近和窗口末尾：
新选择索引重建的mask与原mask逐位相等；注意力输出相对差最大约1.82e-5。

8k整模型跨2k配对：NLL差+0.000203，KL=0.000500，64个后续teacher-forcing token的top1全一致。
直接host快照与原克隆后转host方式逐位相等，reset/load后也一致。

优化配置全部通过装配及图捕获：

| 窗口 | dev0层数 / dev1层数 | dev0 allocated | dev0 reserved |
|---|---:|---:|---:|
| 32768 | 42 / 6 | 55.74 GiB | 58.11 GiB |
| 65536 | 41 / 7 | 55.15 GiB | 57.51 GiB |
| 131072 | 40 / 8 | 55.24 GiB | 57.68 GiB |

启动成功不等于完整长文验收通过；后续实测见下节。

## 完整请求协议

每档输入为窗口减128 token，输出上限64 token，三个新事实分别放在约10%、50%、90%深度。
服务重新加载，只有静态模板前缀可复用；测量HTTP提交至首token、完整输出、实际prefill吞吐。
另测相同prompt缓存复用。记录事实检索结果，不以HTTP 200代替正确性。

| 窗口档位 | 实际输入 | 实际输出 | Prefill | Prefill吞吐 | 首token | HTTP端到端 | 三事实检索 |
|---|---:|---:|---:|---:|---:|---:|---|
| 32768 | 32640 | 45 | 68.884 s | 473.78 tok/s | 69.797 s | **73.822 s** | 3/3 |
| 65536 | 65408 | 45 | 141.595 s | 461.91 tok/s | 145.062 s | **149.395 s** | 3/3 |
| 131072 | 130944 | 44 | 289.236 s | 452.71 tok/s | 295.310 s | **299.961 s** | 3/3 |

三档均出现`LONG_ACCEPT`，均完成全部prompt而非仅启动或设置靠后的位置。
每档的200-token质量基准相同：NLL2.164792、PPL8.7128、top1=48%。
相同prompt重复请求均新prefill为0，三事实再次全部正确。
以上为服务已加载、服务器本机HTTP测量，不含模型冷启动和外部网络传输；各档单次新prompt。
三事实检索属于本次输入的功能检查，不代表所有长文任务均能达到同样准确率。

长测初次启动曾因工作目录使`python -m fnx.server`加载旧包，已纠正为隔离目录后完整重跑；
上表只取纠正后新实现的验收结果。

## 最终运行状态与切换

已将验证后的三个模块部署至`/root/ascend-dev/fnx/fnx`，默认窗口**131072**，
`FNX_LPC=40 FNX_QSA_SPARSE=1`，仍为48层模型、512-token prefill图。
`fnx.service` active/enabled；`/health`返回ctx=131072、qsa_sparse=true。
部署后新短请求“3加5”回答8，服务正常。

四档配置写在`/root/ascend-dev/r52/active_profile.env`；可以持久切换并重启服务：

```bash
bash /root/ascend-dev/r52_switch.sh 8k
bash /root/ascend-dev/r52_switch.sh 32k
bash /root/ascend-dev/r52_switch.sh 64k
bash /root/ascend-dev/r52_switch.sh 128k
```

128k档静态前缀host快照约3567 MiB，新对话首次分配缓存会有额外开销：
部署后首次21-token短输入、输出2 token，端到端5.589 s，其中prefill0.487 s。
低延迟短问答可以使用8k档；大窗口也会增加host缓存占用。

## 运维与回滚

8k起点备份`/root/ascend-dev/r52/baseline.tgz`，环境`r52/baseline.env`。
隔离源包`/root/ascend-dev/r52/fnx`；没有修改8k生产启动脚本来完成以上实验。
日志位于`/root/ascend-dev/logs/r52_*.log`，结果位于`/root/ascend-dev/r52/result_<ctx>.json`。
实验脚本退出时停止候选服务并重新启动原systemd服务。
