# r160：片上 INT4→BF16 查表的数值与独占时序

2026-11-14，起点e44ed0a，原两张910B4-1，当前物理3/5；单权重组件，不构建第二份完整模型。

**新查表实现与r158逐位一致，但组件时序更慢，拒绝进入整层或整模型；正式继续r152，没有新TPOT结果。**

## 实现与数值

对每个64行K分组和输出列tile，先在UB内生成16个INT4取值×scale并舍入BF16；将原权重解包为索引后Gather这些BF16值，再沿用r158的FP32乘激活、分组归约树和原对角epilogue。没有常驻整模型LUT，没有重新打包权重，也没有把BF16舍入删掉。

tile128/256各有16×tile个BF16表项，分别占4/8KiB；对应FP32建表暂存8/16KiB。表和索引仅在UB存在，每个分组重新构造，不据此宣称全部UB占用只有这个表大小。

两卡分别用真实L00/L24 down；每次53组输入覆盖全部512专家、重复专家511与四档幅度。down输出及完整merge输出均与r158逐位相同。另改变激活、专家ID、路由权重、shared、gate，fresh graph/eager及对r158结果均逐位相同。在线数值阶段不采时序；独占计时前又完整复验一次。

这证明本探针覆盖下的r158数值等价，不是与r152全模型逐位等价，也没有证明旧18k回归已修复。两卡每个进程allocator上限1/32，峰值allocated均515901440字节（约492MiB）；不是物理HBM峰值或流量。

## 完整 down+epilogue 的独占时序

| 逻辑卡 / 层 | 生产 µs | 旧直接down µs | r158舍入 µs | LUT128 µs | LUT256 µs |
| --- | --- | --- | --- | --- | --- |
| 0 / L00 | 60.824 | 49.522 | 69.275 | 173.447 | 263.558 |
| 1 / L24 | 60.910 | 49.600 | 69.355 | 173.834 | 264.114 |

卡0最佳查表版为lut128，耗时是生产路径的2.852倍；16轮中16轮更慢。

卡1最佳查表版为lut128，耗时是生产路径的2.854倍；16轮中16轮更慢。

五路同批比较，每张NPUGraph含64个专家bank调用，每轮replay两次，共128次down+epilogue；16轮正反交替测序。专家bank覆盖全部512权重，不反复仅命中10专家。NPU Event区间含图内执行及间隙，不等于纯kernel求和，也不是HTTP TPOT。两个正式Engine已暂停，计时不与正式模型共享设备。

本实现为保持BF16权重舍入增加了建表、整型转换、偏移计算与Gather。这些改动的净成本超过减少的乘法/转换；尚无PMU分项证据，不能把差值全归因于Gather吞吐、UB bank冲突、HBM或L2。小型片上表不必然比连续算术链快。

## 时间范围与恢复

用户在r159结束后要求继续；本轮短组件实测继续约束于已授权的原小时：00:58:44开始、01:38:44触发独立恢复、01:41:44 guardian备用、01:58:44硬截止。r160未重置起点或延长截止。新增短阶段从01:35:41左右开始，只做两卡组件，完成后立即由独立service恢复原入口。

最终恢复距原小时起点2567.6秒，距r160短阶段开始349.8秒，在原截止内。45个正式文件及p275未改，原容器保持；r152 HTTP、manager、单watchdog、fresh8k原回答一致及临时ENV/lease/drop-in/timer/实验进程清理均验证。正常boot-restore保留。

## 下一步依据

当前直接舍入版与本查表版都不作为整模型候选。要继续自研down，需改变数据通路和计算组织，先在同一真实shape证明净收益；不能仅因算式减少就进入全模型。生产ExpW4.apply已调用W4A16 Cube，简单将现有接口改成三维batch也受产品约束，项目r50已有相关记录。

针对更大TPOT收益，后续可优先将现有W8A8纳入r153的V4验证器实测，并补齐drafter、head、IPC和回滚成本；不把M1的节省从旧V4代理直接相减。完整周期与接受率尚未测，不承诺MTP收益。W8A8单独版本仍需完整质量验收。

接口核实参考：[Cast的A2类型组合](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0073.html)、[Gather](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0092.html)、[CreateVecIndex](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0090.html)、[WeightQuantBatchMatmulV2产品约束](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha001/API/aolapi/context/aclnnWeightQuantBatchMatmulV2.md)。是否能跑仍以本机编译与设备测试为准。

公开证据不含权重、完整进程ENV、原始任务/文本/token IDs或凭据。
