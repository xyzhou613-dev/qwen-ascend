"""Render the completed measurement only after verified production recovery."""
import datetime,json
from pathlib import Path
P=Path(__file__).resolve().parent
a=json.loads((P/'analysis.json').read_text());v=json.loads((P/'final-validation.json').read_text())
assert v['status']=='passed'
for b in (1,4):assert a[f'nd_batch{b}']['status']=='completed'
def cst(epoch):return datetime.datetime.fromtimestamp(epoch,datetime.timezone(datetime.timedelta(hours=8))).strftime('%H:%M:%S')
w=json.loads((P/'window.json').read_text());rest=json.loads((P/'restore-complete.json').read_text())
lines=['# FNX 真正四路并行整模型实测（r164，2026-11-14）','',
'已测到真正 B=4：四路独立会话在同一整模型前向中共同计算，每路实际输出 128 token。正式服务已恢复 r152，实验路径未上线。','',
'| 输入 token/路 | 批量 | 组数 | 每路 token/s | 合计 token/s | 每路平均 TPOT / ms |',
'|---|---:|---:|---:|---:|---:|']
for ctx in ('8000','261900'):
    for b in (1,4):
        s=a[f'nd_batch{b}']['summary'][ctx]
        lines.append(f"| {s['input_tokens']:,} | {b} | {s['samples']} | {s['per_row_tps_median']:.3f} | {s['total_decode_tps_median']:.3f} | {s['tpot_ms_median']:.3f} |")
lines+=['',
f"同实验路径的 B4/B1 总吞吐分别提高 {a['matched_path_comparisons']['8000']['total_throughput_gain_pct']:.2f}% 和 {a['matched_path_comparisons']['261900']['total_throughput_gain_pct']:.2f}%。每路 TPOT 同时增大；这不是单会话延迟提速。8k 为三组中位数，长文各一组，长文没有稳定性区间。",'',
'正式 r152 的 r163 历史单路参考约 30.267/29.760 token/s（8k/长文）。本轮 B1 使用不同执行路径和合成输入，只有约 16.675/12.692 token/s。因此上表的 B4/B1 倍数不能套到正式服务；当前 B4 原型总吞吐在 8k 略高于该历史参考，在长文低于该参考。','',
'## 真实并行的证据','',
'完整 48 层按 24/24 分布在双卡。每个 decode step 一次投影处理四行，MoE 的一次 selected grouped matmul 处理 40 个专家项，QSA、DeltaNet、卷积、KV、位置及 head 各保留四行独立状态。CPU 只在一次批量采样返回后记录四个输出，没有循环四次完整模型前向。','',
'每组四个输入摘要及预填后的隐藏状态摘要均不同；每路均生成 128 token，128 步均有四路输出，四路解码区间六对全部重叠。L00/L03/L24/L27 的 B4 输出与同路径分别执行逐位一致；单路扰动不改变另外三路输出和状态，交换行顺序及新输入图重放检查通过。完整模型在小/大上下文容量下，两个新输入位置的 eager/graph logits、隐藏状态、完整 FP32/卷积状态及已写 KV/index 前缀检查，以及独立 head 图检查通过。','',
'本轮是私有固定批处理整模型运行器，尚不是 HTTP 连续批处理调度器。正式 HTTP 服务仍保留 r152 的原调度。四路在批维内共同计算；两个卡段仍按层序交接，没有实现多个微批的两卡流水重叠。','',
'## 输入、配置和计时口径','',
'输入为重复合成文档及四个不同编号的问题，实际长度分别为 8,043 和 261,943 token。四路共享文档前缀，只预填一次，再复制成四份状态并处理不同后缀。该 TTFT 不代表四份不同长文的冷启动。最多输出 128 token，两个正式停止 token 均按 EOS 处理；本轮所有计入组都输出了 128 token。','',
'decode 总速率 = Σ(每路输出数−1) / 四路首至末输出时间区间的并集；每路速率 = (输出数−1) / 该路首至末输出时间。TPOT 是其倒数。预填、首次输出和排队不混入 decode 速率。时间线由同一次四行采样返回记录，不能解释成四个 HTTP 客户端的网络时间线。','',
'实验使用单进程双设备 FnModel 和 host 中转，保留正式 W4 权重、BF16 dense/head；未采用被 r162 拒绝的 A8W8 head。部分 B1 专用融合算子在该原型中关闭，decode MoE 改用 selected GMM，数值路径有变化。没有执行完整 NLL/任务质量评测，不能部署。','',
'| 模型预填计时（不含分词与 reset） | B1 / s | B4 / s |',
'|---|---:|---:|']
for ctx in ('8000','261900'):
    one=a['nd_batch1']['summary'][ctx];four=a['nd_batch4']['summary'][ctx]
    lines.append(f"| {one['input_tokens']:,} token | {one['ttft_s_median']:.3f} | {four['ttft_s_median']:.3f} |")
lines+=['',
'## 中途失败、修正与局限','',
'初版 scatter 写入的 B4/8k 三组总速率为 30.426/30.445/30.478 token/s；扩大到 256k 时，在图捕获阶段 OOM，单次 KV 写入申请约 1.03 GiB 临时空间，未得到初版长文速度。按恢复脚本提前恢复 r152，保留全部初版结果。','',
'随后改用原地 scatter_nd_update，仅写命中行。2.375 GiB 的四路完整缓存探针在 1/24 单进程显存限额内通过：与小尺寸 scatter 参考逐值一致，新位置/新输入图写入一致，额外峰值 16.019 MiB。修正版完成两种上下文的完整 B4/B1 测试。预填 chunk 同时从 512 改为 2048；版本间的耗时变化不能严格全归因于写入算子。','',
'续跑第一次遇到已被 systemd 回收的安全守护 unit，启动器捕获异常并保持正式服务；随后重新创建守护，沿用原绝对截止继续。跨设备旧图释放出现 EE1010 context 警告；后续新图的 eager/replay 检查通过。旧图释放的设备上下文仍需修正，本轮不把该原型称为生产验收通过。','',
'本轮未新增 HBM/L2、Cube/Vector 或 IPC 计数器，不能据这些 token/s 对硬件瓶颈占比下定论。依据代码可以确认：selected GMM 仍是每路 10 个、共 40 个 M=1 的专家组，尚未把跨请求相同专家合并为多行组；批量变大没有自动改善专家 GEMM 的 M 维。','',
'后续优先验证保留 B1 原生快路径的四状态、多微批流水重叠，并统计路由后按专家合并；再适配 B4 的原生索引/注意力及 DeltaNet 融合。这些方向需要独立实测，当前数字不支持承诺四倍吞吐。','',
'## 恢复与交付','',
f"本轮授权从 {cst(w['start_epoch'])} 开始，第 40 分钟恢复定时为 {cst(w['restore_trigger_epoch'])}，硬截止 {cst(w['hard_cap_epoch'])}。两次实验沿用相同起点，没有重置期限。最终 {cst(rest['epoch'])} 恢复正式 r152，距起点 {rest['elapsed_s']:.1f} 秒（{rest['elapsed_s']/60:.2f} 分钟），在 60 分钟内。",'',
f"恢复复验通过：{v['production_files_unchanged']} 个正式文件和 p275 摘要未变，原容器保持，manager、单 watchdog 和正常开机恢复正常；新鲜 8k/128 输出与既有 r152 回答完全一致。临时 drop-in/timer、lease、私有环境和实验进程已清理。",'',
'原始数据与复算：`batch4-results.json`、`batch1-results.json`、`analysis.json`；初版失败保留于 `attempt-scatter/`。组件校验、图检查、原/补充冻结清单、授权和绝对时限、恢复复验及脚本一并交付。生产代码没有集成该实验路径。','']
(P/'report.md').write_text('\n'.join(lines))
print('R164_REPORT_RENDERED')
