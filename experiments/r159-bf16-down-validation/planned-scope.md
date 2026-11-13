# r159：BF16 舍入修正的分级实测

只准备私有 runtime，不修改正式 r152 源码或 p275。r158 的已编译库保持原 SHA，新增 wrapper 将 down 调用接到 fnx_w4_bf16w，epilogue 沿用 r156 库；本轮没有声称整模型逐位一致。

1. CPU preflight：编译全部 Python、检查 shell 语法、冻结 runtime/脚本/输入哈希，并确认当前生产 45 文件、入口和健康状态。此步骤不中断服务，不初始化 NPU。
2. 新的有界窗口内，先暂停已识别的原 Engine、manager 和 watchdog；每个组件和模型启动前拒绝其他 Engine 或任何 NPU 占用。
3. 两卡各取真实 L00/L24 down：生产路径、旧 down、BF16 修正版 tile128/256 的完整 down+epilogue。53 组输入覆盖 512 专家、四档幅度及重复专家；64 banks graph 时序覆盖全部权重银行，16轮正反交替顺序，另测更新全部输入后的 graph/eager。仅按组件耗时选择 tile；两卡均需比生产快至少2%。
4. L00/L03/L24/L27 实际层三路比较，检查输出误差、conv/FP32 递推或 KV/index 状态。修正版 relative L2 上限 .001，且四层均须快于 W8A8 原 down，才进入整模型。
5. r152 与 W8A8+BF16修正各加载一次，独立进程图捕获审计确认48层/96投影及双进程实际加载修正版库。先做旧五切片真实 prefill、4个冻结NLL样本（2048目标）和原10个自然文本片段，再 fresh8k/262016 各三次、缓存和持续输出。输入复用r157，基线旧五切片需复现到5e-5内。

这些是定位回归和性能的短筛查，不是完整质量验收。r157 的1200任务、100352目标及域级置信界门槛不修改。旧18k现已用于修正开发，不能称为新的未见验证集。本轮不补齐旧候选的201基线题，也不把r157约18%收益自动转记到修正版。

计划窗口最多60分钟：40分钟绝对 timer 触发独立恢复 service，43分钟独立 guardian 备用，主 controller 最长41分钟。失败/正常退出都调用独立恢复 unit，SSH断开仍执行；不重置 window-start。恢复原入口、manager、单 watchdog，清除本轮私有ENV/lease/drop-in/timer，fresh8k确认原回答，最后逐项校验45生产文件未变。正常 boot-restore 保留。

`prepare.py`仅可执行一次；`preflight.py`在窗口前冻结。`start_window.py`只在本轮停服范围已获授权后执行。不得在生产模型占卡时启动第二份完整模型。未使用的时间不自动扩展为完整质量长跑。
