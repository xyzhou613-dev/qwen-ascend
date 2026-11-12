# r150 固定归约顺序的 q/k 全融合：正式部署与复测

2026-11-12，192.168.18.206，2×Ascend 910B4-1（物理3/5，逻辑0/1，24+24两进程），CANN9.1.0、驱动26.1.1、torch2.10.0+cpu、torch_npu2.10.0.post4。本轮起点为r144实现794a22e，仓库HEAD cacad4d且干净。

正式启用 `FNX_NORM_ORDERED=1`，由原p275.sh入口加载，健康版本r150。两个场景均通过精度与性能门禁，并在正式入口再次复现收益。此前r148候选因不精确被否决；r149保留CANN归约虽精度通过，但8k仅+0.119%、256k反降0.174%，也未发布。

## 实际TPOT（ms/token）

| 场景 | 同轮r144 | 隔离候选 | 正式r150 | 正式输出速度变化 |
|---|---:|---:|---:|---:|
| fresh 8k（三次中位） | 33.80 | 33.07 | 33.02 | +2.36% |
| fresh 256k（各一次） | 34.51 | 33.70 | 33.66 | +2.53% |
| 缓存256k（各一次、非流式） | 34.12 | 33.37 | 33.31 | +2.43% |


输出速度变化为`TPOT基线/TPOT新版本-1`。8k输入8064、输出128，每版本三份固定输入取TPOT中位；256k输入262016、输出128，每版本一次，不能外推长输入追加8k输出。缓存重复新增prefill=0，且使用非流式接口，单列展示，不冒充fresh收益。

HTTP TTFT：8k中位 3.846→3.846s，256k 69.740→69.846s。该改动针对decode，**没有确认长文TTFT改善**，不能把单次微小差异解释为prefill提速，也未实现最初20%的输出加速目标。

基线、候选各冷启动，隔离在127.0.0.1:19002；原FNX、watchdog和管理器在受控维护流程内停止，每次只有一个Engine，使用同配置、权重和冻结输入。独立限时恢复任务保护断链场景。HTTP ready前的图捕获/启动时间不混入TTFT。隔离A/B后先恢复r144，再安装仅五个受控文件并通过正式入口重测。

## 为什么这一版能修复旧反例

原r144是两次BF16平方、CANN ReduceSum、查表后处理，加query缩放，每个线性层7个kernel。新路径在固定单token `[1,10240]` 卷积/SiLU输出上用1个kernel同时处理q/k，36层减少约216个kernel/token。BF16平方舍入、归约到BF16、原CANN生成的epsilon/Rsqrt查表、归一化BF16舍入和query缩放舍入均保留。

r148的原生128元素归约与CANN累加顺序不同。新版本先将两个64元素片段逐元素相加，再对64元素归约，修复四个已保存的稀疏边界反例。该顺序是在本机CANN9.1、固定R=128形状上对拍得到的，不证明其他CANN版本或任意形状同序。不同batch、dtype、shape或非连续输入回到原路径；`FNX_NORM_ORDERED=0`可停用新分支。没有改FP32递推、MatMul、prefill BMM或模型权重。

## 精确性检查

双卡各4608组随机/SiLU/原始BF16编码和稀疏边界输入，以及四个历史反例均逐位通过。全部36个真实权重DeltaNet子路径，逐层16次eager+8次动态graph replay，共864例，输出、BF16卷积历史与FP32状态逐位相同。完整L00/L12/L24/L44的decode与128/512/2048 prefill也逐位通过。组件图事件中位约16.8→9.1µs，整模型收益以上表为准。

| 冻结文本 | token数 | 目标数 | 两版本平均NLL | top1 | NLL/argmax对照 |
|---|---:|---:|---:|---:|---|
| natural6000 | 6000 | 199 | 1.722880 | 0.6131 | 全部逐位相同 |
| service12000 | 12000 | 199 | 0.000958 | 1.0000 | 全部逐位相同 |
| technical24000 | 24000 | 199 | 1.127045 | 0.7387 | 全部逐位相同 |
| natural18000 | 18000 | 199 | 3.512499 | 0.4020 | 全部逐位相同 |
| natural36000 | 36000 | 199 | 3.149386 | 0.3518 | 全部逐位相同 |


五份共995个尾部目标，使用真实 `plan_prefill/prefill_chunk/finish_prefill`，不使用逐token loss接口代替prefill验证。新增18k/36k为鲁迅自然文本，来源为维基文库《阿Q正传》及《彷徨》篇目，按清单抽取段落，只连接一次并取不重叠token切片，没有重复填充；URL、HTML/文本/ID哈希见quality-input-manifest.json。原12k服务文本仍是重复样本，不能单独代表长文质量。样本通过不构成所有输入等价的证明，历史100题未重跑。

隔离三份fresh 8k完整回答、256k完整回答和缓存回答相同；正式入口再次全部回答精确相同，256k三个事实召回3/3。保留所有逐目标NLL、argmax、HTTP统计与冻结输入。

## 部署、回退与复现

新增 `fnx/norm_ordered.py`、`kernels/l2norm_ordered/`，改engine的单token分支、health版本和启动开关。Linux aarch64 ELF在部署证据implementation/内，仓库保存源码与Bisheng构建脚本：`bash kernels/l2norm_ordered/build.sh /root/ascend-dev/fnx/fnx/libfnx_norm_ordered.so`。复用已有norm_post查表；需部署相同版本的Python/ELF、开启开关后重启以重新捕获图。

启动参数补丁为 `python3 scripts/r150-enable.py /root/ascend-dev/p275.sh --check`，去掉--check才写入；可重复检查。回退优先恢复deployment-backup中五个文件的原内容并由原入口重启，或设置FNX_NORM_ORDERED=0后重新捕获图。不会把诊断质量路由放到正式server.py。

最终健康检查、五个部署文件哈希、正式HTTP复测和管理器/watchdog状态均已验收。临时租约、恢复timer、drop-in和完整ENV快照已移除。原始实验目录 `/models/fnx-diagnostics/r150-ordered-norm-20261112`；证据采用白名单打包，不含完整ENV、权重、图缓存或用户请求激活。

完整证据见[证据目录](../benchmarks/r150-ordered-norm/README.md)。
