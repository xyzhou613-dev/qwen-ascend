"""CPU-only analysis of isolated TPOT direction probes."""
import json,statistics
from pathlib import Path
P=Path(__file__).resolve().parent
def read(name):return json.loads((P/name).read_text())
def mean(x):return statistics.mean(x)
baseline=32.89
dense=[]
for name in ['qkv','z','qkv_z','out','full_q']:
    variants={}
    for label,prefix,quant in [('ND','dense-stream','w8a8_t'),('NZ','dense-stream-nz','w8a8_nz')]:
        rows=[next(r for r in read(f'{prefix}-device{d}.json')['records'] if r['name']==name) for d in range(2)]
        variants[label]=dict(bf16_us=mean(r['timing']['median_us']['bf16'] for r in rows),
            w8a8_us=mean(r['timing']['median_us'][quant] for r in rows),
            relative_l2_mean=mean(e['relative_l2'] for r in rows for e in r['errors']),
            relative_l2_max=max(e['relative_l2'] for r in rows for e in r['errors']),
            formats=[r['layouts'] for r in rows])
    dense.append(dict(name=name,shape=rows[0]['shape'],variants=variants))
def timing(name,variant,kind):return next(r for r in dense if r['name']==name)['variants'][variant][kind+'_us']
dense_save=(36*(timing('qkv','NZ','bf16')+timing('z','NZ','bf16')-timing('qkv_z','NZ','w8a8'))
            +48*(timing('out','NZ','bf16')-timing('out','NZ','w8a8'))
            +12*(timing('full_q','NZ','bf16')-timing('full_q','NZ','w8a8')))/1000
selected_params=36*16384*2560+48*2560*6144+12*12288*2560
verify={}
for variant in ('small','loop','native'):
    records=[r for d in range(2) for r in read(f'verify-layer-{variant}-device{d}.json')['records']]
    type_times={kind:{str(T):mean(r['median_ms'][str(T)] for r in records if r['type']==kind) for T in (1,2,4)} for kind in ('linear','full')}
    verify[variant]=dict(type_times=type_times,weighted_48_layer_proxy_ms={str(T):36*type_times['linear'][str(T)]+12*type_times['full'][str(T)] for T in (1,2,4)})
hccl=[dict(rank=d,records=read(f'hccl-rank{d}.json')['records']) for d in range(2)]
assert all(r.get('dynamic_graph_sum_verified') for d in hccl for r in d['records'])
mtp=read('mtp-inventory.json'); ep=mtp['expert_parameters'];dp=mtp['total_parameters']-ep
mtp_memory=dict(dedicated_params=mtp['total_parameters'],bf16_GiB=mtp['total_bf16_bytes']/2**30,
    W4_g64_plus_BF16_dense_estimate_GiB=(ep/2+ep/64*2+dp*2)/2**30,
    shared_embedding_and_head_inclusive_params=mtp['total_parameters']+2*248320*2560,
    standard_kv_GiB=2*2*262144*256*2/2**30,
    packed_kv_GiB=(512+64)*2*262144*2/2**30,
    index_raw_and_pooled_GiB=(262144*128*2+262144/4*128*4)/2**30,
    four_token_verifier_five_prefix_delta_journals_MiB=36*5*48*128*128*4/2**20)
cost=[dict(p=p,k=3,expected_emitted=sum(p**i for i in range(4)),
    max_cycle_ms_for_20pct_speedup=baseline/1.2*sum(p**i for i in range(4)),
    max_cycle_ms_for_50pct_speedup=baseline/1.5*sum(p**i for i in range(4))) for p in (.7,.8,.9,1.)]
result=dict(status='exploration_complete_not_deployed',production_revision='r152',baseline_http_TPOT_ms=baseline,
    target_ms_for_20pct_speedup=baseline/1.2,target_ms_for_20pct_TPOT_reduction=baseline*.8,
    dense=dense,dense_weighted_component_saving_proxy_ms=dense_save,
    dense_selected_storage_saving_GiB=selected_params/2**30,
    verify=verify,hccl=hccl,mtp_memory=mtp_memory,mtp_cost_gate=cost,
    full_model_candidate_measured=False,quality_regression_run=False,
    precision_policy='User permits numerical changes subject to broader model-quality evaluation.')
(P/'summary.json').write_text(json.dumps(result,indent=2))

table=[]
names={'qkv':'DeltaNet qkv','z':'DeltaNet z','qkv_z':'qkv+z 合并','out':'attention out','full_q':'全注意力 q'}
for r in dense:
    v=r['variants']['NZ']; nd=r['variants']['ND']
    table.append(f"| {names[r['name']]} | {v['bf16_us']:.2f} | {nd['w8a8_us']:.2f} | {v['w8a8_us']:.2f} | {(1-v['w8a8_us']/v['bf16_us'])*100:.1f}% |")
vt=[]
for variant,label in [('small','固定活跃组 GMM'),('loop','逐 token MoE'),('native','逐 token MoE + 小 T FP32 recurrence')]:
    v=verify[variant]['weighted_48_layer_proxy_ms']
    vt.append(f"| {label} | {v['1']:.2f} | {v['2']:.2f} | {v['4']:.2f} |")
ht=[]
for index,label in enumerate(['H=2560 BF16 / 5 KiB','HCH=10240 BF16 / 20 KiB','H=2560 FP32 / 10 KiB']):
    a,b=[d['records'][index] for d in hccl]
    ht.append(f"| {label} | {a['median_graph_us']:.2f} | {b['median_graph_us']:.2f} | {max(a['median_graph_us'],b['median_graph_us'])*.096:.2f} |")
ct=[]
for r in cost:
    ct.append(f"| {r['p']:.1f} | {r['expected_emitted']:.3f} | {r['max_cycle_ms_for_20pct_speedup']:.1f} | {r['max_cycle_ms_for_50pct_speedup']:.1f} |")
report=f'''# r153：大幅降低 TPOT 的方向验证

2026-11-13，192.168.18.206，双 Ascend 910B4-1，物理 3/5、容器逻辑 0/1。生产仍为 r152，基线提交 db5dc7e7f776fbd28fb8ab936ee1c416ceeda7f1。r153 是实验编号，没有发布新服务版本。

用户已明确允许数值变化，条件是通过更完整的质量评测。本轮没有停服、重启或启动第二个完整 Engine；只运行限内存、限时的权重/单层/双 rank 通信探针。所有数值路径改动均在诊断目录；尚无新的整模型 TPOT 或质量验收结果。

结论：下一阶段应研究“大投影 W8A8 + 融合 MoE GEMV”，同时把 MTP 的第一里程碑设为足够快的批量验证器。W8A8 单项实测支持继续实现，但所覆盖投影的加权代理只节省约 {dense_save:.2f} ms，不能承诺独自达到 20%。MTP 上限更高，但直接接现有 prefill 会更慢。TP2 的通信已可图捕获，跨 NUMA 的逐层通信成本使其收益不确定，暂列备选。

## 1. 目标及旧 profile 的边界

r152 正式 HTTP 8k TPOT 中位数 32.89 ms，长文 33.61 ms，取自 r152 已完成验收，本轮未重新建立端到端基线。输出速度提高 20% 需达到 **27.41 ms**，节省 5.48 ms；TPOT 降低 20% 的目标是 **26.31 ms**，两者不同。

r145 对 r144 的旧采样：8k 周期 34.143 ms，设备时间并集 31.720 ms，无跨卡 decode 重叠；dense/index MatMul 11.937 ms、量化 MatMul 6.419 ms、expert_gather 2.450 ms；约 2.42 ms 是设备并集外的间隙，不能全归为 Host 调度。r150 以后又消减了一些 kernel，因此这些仅用来定位大方向，不是 r152 剩余占比。本轮不声称补采了 r152 PMU 或最新完整 roofline。

“总 HBM 带宽没打满”不能推出“所有 kernel 都不是访存限制”：流水线每卡等待另一阶段、大量短 kernel 的固定成本和部分大投影的流式访存瓶颈可以同时存在。不能继续用整卡平均带宽低来预测任意 GEMV 都能数倍提速。

## 2. W8A8：真实权重、流式工作集与 NZ 格式

取 L00/L24、L03/L27 真实稠密权重，按生产 DenseW8.deq_w 路径恢复 BF16，再逐输出通道量化 int8。输入为 16 组、4 种幅值的合成 BF16 激活。每次调用含 npu_dynamic_quant 的 per-token 量化，未将量化成本移出计时。不是原始 BF16 模型的新量化质量验收。

每个变体轮换至少 384 MiB 权重副本，64 次调用/图，12 轮交替顺序，各两次 replay，事件时间均摊每调用；两卡依次运行。该方法降低单权重热缓存误导，但没有采 PMU 来证明每次访问都穿透 L2，因此称“流式代理”。不将其直接视为整模型耗时。

下表单位 µs，为两卡各自中位数的均值。ND 与 NZ 来自不同轮，同轮都有 BF16 参考；格式 29 在所有 NZ 副本上实际检查通过。

| 投影 | BF16（NZ 同轮） | W8A8 ND | W8A8 NZ | NZ 相对同轮 BF16 耗时下降 |
|---|---:|---:|---:|---:|
{chr(10).join(table)}

当前生产 qkv 与 z 是两次投影，合并+NZ 更合理的对比是两者合计约 80 µs → 42 µs。按 36 次 qkv+z、48 次 out、12 次 full-q 加权，代理节省约 **{dense_save:.2f} ms/token**。权重存储约省 **{selected_params/2**30:.2f} GiB（两卡合计）**，不含 padding、scale 和工作区，不能把两卡之和当成单卡可用空间。小 mixer 的热缓存测试 W8A8 反而更慢，不做全面替换。

这批合成激活的输出相对 L2 误差约 1.6%～2.2%，不等于准确率下降比例，也不能作为上线门槛。下一版应以原始 BF16 权重和独立校准集研究 per-channel / outlier 平滑，router、norm、FP32 递推先保留原精度，敏感层允许回退 BF16。质量判断看完整任务，而不是只看余弦或 argmax。

NZ 探针最初因隔离 Python 环境缺 decorator/scipy/attrs/psutil、NumPy long 别名失败，不能误判为硬件不支持。最终仅在诊断私有路径补依赖与 compiler 子进程 sitecustomize，生产 venv 和启动环境未改。SciPy 对私有 NumPy 版本仍有兼容警告；NZ 最终算子执行、数值对照和图重放均完成，但私有补丁不直接纳入生产依赖方案。

## 3. TP2：通信可行，但要付逐层成本

npu-smi 拓扑确认物理 3/5 为 SYS：经过 PCIe 和跨 NUMA 互联，无 HCCS。HCCL 两个小辅助进程，BUFFSIZE=8 MiB、CPU 分别绑到对应 NUMA；先验证两 rank 求和，再捕获 96 次同缓冲 all-reduce。图重放用新填入的非零数据验证 96 次求和结果，避免只测零输入而未验证通信。未与计算重叠，未引入生产模型。

| 消息 | rank0 图内 µs/次 | rank1 图内 µs/次 | 按较慢 rank 的 96 次合计 ms |
|---|---:|---:|---:|
{chr(10).join(ht)}

这是裸通信图的代理，含该测试的进程发射/同步影响；不是已实现 TP 的整层成本。重复图内 HCCL 约 46～51 µs 量级，eager 约 0.18～0.20 ms/次。标准 attention-out + MoE-down 各一次 all-reduce、48 层约 96 次，对应约 4.5～5 ms。当前 PP 只有一个阶段交接，不能用其 0.72 ms 旧空档代替 TP 的通信预算。

TP 会复制 router、HC、norm 等路径，不能把整个 32.89 ms 除二。应先做一层真实权重分片的含通信对照：DeltaNet heads 16/48→8/24、QSA heads 24/2→12/1、专家中间维 640→320；包括 W4 重打包和对应 scale。只有含两个 all-reduce 的整层时延足够低，再推进全 48 层；已有通信图不能证明这个门槛通过。

## 4. MTP：先验证成本，再加载 drafter

原始 safetensors 头部确认 31 个 mtp.* 张量、1 层 full-attention MTP。**增量专用参数为 {mtp['total_parameters']:,}（2.607B）**，BF16 {mtp_memory['bf16_GiB']:.3f} GiB。若把共享 embedding 与 lm_head 也计入模块规模，约 3.879B，接近“4B”的说法；这两种统计口径不同，新增 HBM 应按增量权重计算。

专家部分占 2.517B。若重新量化为 g64 W4、BF16 scale，其他专用权重保留 BF16，权重约 **{mtp_memory['W4_g64_plus_BF16_dense_estimate_GiB']:.3f} GiB**，仅是尚未生成 artifact 的估算。262144 窗口的原始 K/V 约 0.500 GiB；当前 native packed K/V 表示另需 0.563 GiB，raw/pooled index 合计 0.094 GiB。若照搬生产 dual-cache，则缓存合计约 **1.156 GiB**；若能仅保留 packed 路径，则约 **0.656 GiB**。还需 embedding、图工作区和回滚空间，不能只凭权重估算宣称一定放得下。

根据 [vLLM 的 Qwen4Exp MTP 源码](https://docs.vllm.ai/en/latest/api/vllm/models/qwen4_exp/nvidia/mtp/)，输入连接应取主模型最终 mixer **之前的 4 路 HCH=10240 hidden**；fc_hidden 按分支共享，另加 token embedding 的 fc_embedding。输出同时保留 collapse 后 H=2560 给 lm_head，和下一草稿所需的多路 hidden。不能直接拿最终 norm 后的 H=2560 替代。现装 transformers 参考忽略 mtp.*，FNX 没有现成 MTP forward。

MTP 的前缀 QSA KV 也要初始化；应随着目标模型 prefill 分块消耗 HC hidden，避免为整个 256k 前缀额外保留约 5 GiB 的 hidden。stage1 还需可由设备草稿 token 索引的 embedding，不能误用不绑定的 lm_head 当 embedding，也不应每个草稿都回 CPU 读取 token。

本轮已实际实施三个验证器方向：

1. 通用 512-group MoE 在 T=2 时申请 1.58 GiB 临时区，触及探针 1/32 allocator 上限（当时另有 3.14 GiB 物理空闲），并非生产模型 OOM。保留失败日志与服务健康记录。
2. 图内选最多 T×10 个可能活跃专家，包括零行 padding 组，避免 host routing 读取和全 512-group 临时区；用真实权重、重复/不同专家测试与 Cartesian 参考对照。门槛数值对照均零误差，但通用小 group GEMM 很慢。
3. 改成逐 token MoE，同时保留批量 attention/HC；进一步将 T≤4 的 DeltaNet 三角求解替换为现有 native FP32 顺序 recurrence，保存每个前缀状态。双卡各 8 组独立 FP32 PyTorch recurrence 对照，输出及所有前缀状态最大差均 0。它未覆盖整个批量层与逐 token 层的数值等价性。

下表将两卡各一个 linear 层与一个 full 层的测量中位数按 36+12 层加权。**这是 4 层合成 hidden 的成本代理，不是执行了 48 层，更不是整模型验证周期。** 当前 private context=8192，position=4096，KV 初态为零；没有 draft、lm_head、IPC、完整 rollback 和长文检验，单层缓存也可能比真实网络更有利。

| 验证器代理 | T=1，ms | T=2，ms | T=4，ms |
|---|---:|---:|---:|
{chr(10).join(vt)}

T=4 的代理从约 131 ms 降至 80 ms，说明专用验证路径有实际优化空间；但 **T=2 没有值得加入 drafter 的成本优势**。T=4 还不能承诺显著提速。当前状态 journal 为 T+1 个 FP32 前缀；T=4 在 36 层合计约 540 MiB，分布在两卡。还须处理 conv 历史、KV/index 尾部、位置与拒绝后的下一步，禁止每轮复制完整 7.76 GB conversation snapshot。

若一轮提出 k=3 个草稿、目标批量处理 4 个位置，在各条件接受率均为 p 的简化假设下，平均产出 E=1+p+p²+p³。实际评估必须以真实每轮产出均值和完整周期计时为准：TPOT=(draft总时长+target验证+回滚恢复)/实际产出。

| p（假设） | E（假设） | 达到输出速度 +20% 的周期上限 ms | 达到 +50% 的周期上限 ms |
|---|---:|---:|---:|
{chr(10).join(ct)}

p=0.8 时，+20% 需完整周期≤80.9 ms；现有 target 组件代理已约 80 ms，还没加 draft/head/IPC，故暂不满足。若 draft+回滚暂按 10 ms 预算，target 应压到约 71 ms；若追求 1.5 倍输出速度，则需约 55 ms。10 ms 是设计预算而非测量。接受率及 drafter 成本尚未测量，不能因含 MTP 权重就宣称 1.5～2 倍。

## 5. 下一阶段排序与停损点

| 顺序 | 方向 | 本轮依据 | 下一项明确门槛 |
|---|---|---|---|
| 1 | 大投影 W8A8/NZ + qkv/z 合并 | 大投影耗时降约 10%～46%，但代理总收益不足 20%；还能释放 HBM | 真实激活校准、单层对照；整模型质量通过后再测端到端，不量化变慢的小 mixer |
| 1 | 专用 MoE W4 GEMV：融合专家选择/加载、反量化、计算 | 旧预算中量化+gather 约 8.87 ms，值得争取数毫秒；现有小批 GMM 路径差 | 先打 gate/up 的 K2560,N1280 与 down 的 K640,N2560、TOPK10，含动态 ID、scale、搬运和加权归并；两卡组件总计至少省 3～4 ms 的目标预算，再上整模型 |
| 2 | MTP + 专用 3～4 token 验证器 | 原型去掉部分通用路径成本，但当前约 80 ms 代理仍偏高 | 先将含 head/通信/状态保存的 V4 压到可盈利预算，再测 MTP 接受率；未过性能门槛不加载完整 drafter |
| 3 | TP2 / W8A8+TP2 | HCCL 图可用，但逐层通信约 4.5～5 ms | 先验证单层 shard+两次 all-reduce 净收益，不能按总耗时/2推演 |

融合 MoE 的“省 3～4 ms”是和 W8A8 组合突破 5.48 ms/token 目标的研发门槛，尚未实现。不能把当前 down Cartesian 的 10×10 逻辑输出误称为 10 倍可省物理 Cube 工作。以往简单 GMM 扫描无稳定收益，这次必须改变权重读取和执行组织，避免重复旧调参。[Ascend W4A16 研究](https://arxiv.org/abs/2601.16536)也将额外 GM 权重搬运列为重要瓶颈；论文的特定 shape 加速不能当 FNX 收益预测。

多请求连续 batching 可提高 aggregate tokens/s，但并不保证单个用户 TPOT 降低，本轮不把它列为已满足用户目标的方案。

## 6. 数值变化的质量验收

冻结 r152 对照，冻结输入、tokenizer、模板、随机种子及长度，校准集与验收集分开。新门槛采用任务质量而非逐位一致；以下是下一候选的预先工程验收方案，**本轮尚未执行完整任务集**。

- 至少 100,000 个独立 held-out 预测目标，覆盖中英自然文本、技术、代码、数学，512/2k/8k/32k 等长度；真实 prefill 与 decode 分开记录。建议平均 NLL 增量≤0.02 nat、各预先定义切片≤0.05 nat；同时保留分布和异常样本，不只看总体均值。
- 至少约 1,000 道独立任务，分开报告数学、代码真实测试 pass@1、中英知识/阅读、JSON/tool-call 结构遵循，不用一次 LLM 自评分替代可执行判定。覆盖原历史 100 题，但不以其代替新增任务。按领域做配对非劣效分析，建议正确率容忍 2 个百分点，报告单侧 95% 区间；区间过宽就加样本，不将“无显著差异”当作等价证明。
- 单独测试 32k/64k/128k/262016 输入，多位置多事实检索、长文摘要/推理及追问，覆盖 fresh、prefix-cache、重置、长→短→新长；扩充 r152 只有 5 篇/995 目标的范围。
- 针对 MTP 增加全接受、首/中/尾拒绝、EOS、近窗口边界、草稿状态恢复和拒绝后追问；greedy 及服务支持的采样参数分别检查。若提供采样，需正确的接受/残差采样算法，不能把 greedy token 比较直接用于任意采样。
- 性能验收再采用相同输入的 r152/candidate/回切，多次 fresh 8k、fresh 262016、缓存路径和持续输出，分别统计 TTFT、TPOT中位/p95、峰值 HBM；质量先过门槛再判断是否发布。

原始证据：/models/fnx-diagnostics/r153-tpot-directions-20261113。交付只取脚本、日志、非私密选定配置、摘要与权重 shape/计数，不含模型权重、图缓存、完整进程 ENV、私有依赖包或 pip 配置/日志。最终运行状态和部署哈希见 final-operations.json。
'''
(P/'report.md').write_text(report)
block=f'''## r153 方向探索更新（2026-11-13，生产仍 r152）

用户允许数值变化，但须更完整质量评测。隔离实测完成真实权重 W8A8/NZ、跨 NUMA HCCL 图内 all-reduce、2/4-token 单层验证器代理；未发布新运行版本，未做新端到端提速验收。

- 大投影 W8A8/NZ 比同轮 BF16 耗时降约10%～46%；qkv+z合并约80→42µs。加权代理节省约{dense_save:.2f}ms，不等于整模型收益，单独不足20%。小mixer量化更慢。
- HCCL图内小消息约46～51µs/次，标准TP2按96次约4.5～5ms；物理3/5为SYS、无HCCS。先测真实shard层，不能将总TPOT除二。
- MTP增量专用参数2.607B，BF16约4.856GiB；共享embedding/head也计入时约3.879B。W4权重约1.414GiB是估算；缓存还需单列packed/dual和index空间。
- T4验证器的4层加权代理约131→80ms；包含小T native FP32前缀state journal，但不含head、IPC、drafter和完整回滚。p=.8时输出速度+20%要求完整周期≤80.9ms，当前尚未证明盈利。优先继续压低验证器成本。
- 下一主线是“大投影W8A8+融合MoE GEMV”，MTP随后做真实收益验证，TP2备选。更改数值路径须补多域NLL、约千题数学/代码/中英/结构任务及多长度长文/追问，不能仅用原995个目标上线。

详见 [r153 方向实测](r153-tpot-directions.md)。r145的r144硬件占比只作历史参考，不作r152剩余预算。

'''
(P/'techdoc-block.md').write_text(block)
print(json.dumps(dict(dense_save_proxy_ms=dense_save,verify_proxy=verify,mtp_memory=mtp_memory),indent=2))
