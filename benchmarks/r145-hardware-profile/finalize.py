"""Finalize measured evidence; CPU-only, never creates an Engine or changes deployment."""
import csv, json, runpy, shutil
from pathlib import Path
import analyze as A

P = Path(__file__).resolve().parent
s = json.loads((P/'summary.json').read_text())
runs = json.loads((P/'runs.json').read_text())
s['traffic_units'] = {
    'system_hbm': 'Exporter MB/s, summed over four channels; MB=1e6 bytes, GB=1e9 bytes.',
    'memory_access': 'CANN logical GM request volume, KB interpreted as 1024 bytes; GiB=2**30 bytes. Not physical HBM traffic.',
    'l2': 'Raw hit/miss-allocate request counts. Direct l2_read/write_bw fields returned zero and are unavailable for bandwidth attribution.',
    'pcie': 'Throughput MB/s; nonpost_latency us. Zero throughput samples cannot resolve the small messages.',
}
extra_notes = [
    'PCIe throughput sample fields are zero in these short decode windows; do not interpret zero as absence of traffic. PCIe latency counters have different units and are not bandwidth.',
    'HBM system totals sum four channels, whereas exported hbm.csv Average is a channel average. Group kernel-duration sums are not an additive closed budget because concurrent tasks can overlap.',
    'All direct Memory-group L2 read/write bandwidth fields were zero. L2 hit/miss request counts and MemoryAccess logical volumes are reported separately; no physical L2 byte rate is inferred.',
]
for note in extra_notes:
    if note not in s['notes']: s['notes'].append(note)
s['logical_traffic'] = {}
traffic = '| 上下文 / 阶段 | L2读命中, 百万次/token | L2读miss_allocate, 百万次/token | GM逻辑读, GiB/token | GM逻辑写, GiB/token |\n|---|---:|---:|---:|---:|\n'
for context in ('8k', '256k'):
    rec = next(r for r in runs if r['tag'] == context+'_timing')
    a,b,n = A.bounds(rec)
    tasks = [row for stage in (0,1) for row in A.taskrows(rec['tag'],stage)]
    memcpy = A.union(A.cut([(x,y) for x,y,typ in tasks if 'MEMCPY' in typ],a,b))/n/1e6
    core = A.union(A.cut([(x,y) for x,y,typ in tasks if typ in A.DEVICE],a,b))/n/1e6
    s['timing'][context]['all_memcpy_union_ms_per_token'] = memcpy
    s['timing'][context]['core_kernel_union_ms_per_token'] = core
    s['logical_traffic'][context] = {}
    for stage in ('0','1'):
        l2 = s['pmu'][context]['L2Cache']['stages'][stage]['total']
        counter_rec = next(r for r in runs if r['tag'] == context+'_l2')
        periods = len(counter_rec['rows'])-1
        f = s['pmu'][context]['MemoryAccess']['stages'][stage]['total']['fields']
        read = sum(f[k]['per_token'] for k in ('aic_read_main_memory_datas(KB)','aiv_read_main_memory_datas(KB)'))/1024**2
        write = sum(f[k]['per_token'] for k in ('aic_write_main_memory_datas(KB)','aiv_write_main_memory_datas(KB)'))/1024**2
        hit,miss = l2['l2_read_hits']/periods, l2['l2_read_misses']/periods
        memory = s['pmu'][context]['Memory']['stages'][stage]['total']['fields']
        l2_fields = {k:v['weighted_mean'] for k,v in memory.items() if '_l2_' in k}
        assert len(l2_fields)==4 and all(v==0 for v in l2_fields.values())
        s['logical_traffic'][context][stage] = dict(l2_read_hits_per_token=hit,l2_read_miss_allocate_per_token=miss,
            gm_logical_read_GiB_per_token=read,gm_logical_write_GiB_per_token=write,direct_l2_bandwidth_valid=False)
        traffic += f'| {context} / S{stage} | {hit/1e6:.3f} | {miss/1e6:.3f} | {read:.3f} | {write:.3f} |\n'

    # Retain all counter points, including both flanks of the integration window.
    hwrec = next(r for r in runs if r['tag'] == context+'_hardware')
    lo,hi = hwrec['rows'][0]['start_ns'], hwrec['rows'][-1]['end_ns']
    srcs = list((P/'system'/context).rglob('msprof_*.json'))
    assert len(srcs)==1,srcs
    src = srcs[0]; raw = json.loads(src.read_text()); events = raw if isinstance(raw,list) else raw['traceEvents']
    names,labels = {},{}
    for e in events:
        if e.get('ph') == 'M':
            if e['name']=='process_name': names[e['pid']] = e.get('args',{}).get('name')
            if e['name']=='process_labels': labels[e['pid']] = e.get('args',{}).get('labels')
    with (P/'aligned'/f'{context}_system_counters_full.csv').open('w',newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(['device','kind','metric','channel','timestamp_ns','value','unit','inside_generation_window'])
        for e in events:
            kind = names.get(e.get('pid'))
            if e.get('ph')!='C' or kind not in ('HBM','PCIe'): continue
            ts = A.ns(e['ts'])
            unit = 'us' if 'latency' in e['name'] else 'MB/s'
            for key,value in e.get('args',{}).items():
                if A.number(value) is not None:
                    writer.writerow([labels[e['pid']],kind,e['name'],key,ts,value,unit,int(lo<=ts<=hi)])
    out = P/'system-summaries'/context; out.mkdir(parents=True,exist_ok=True)
    for pattern in ('hbm_*.csv','pcie_*.csv'):
        for f in (P/'system'/context).rglob(pattern): shutil.copyfile(f,out/f.name)

(P/'summary.json').write_text(json.dumps(s,indent=2))
runpy.run_path(str(P/'report.py'))
text = (P/'report.md').read_text()
intro = '''本轮补采回答：正常runtime基线约33.35 / 34.22 ms；带timing的账本中约93%为设备任务执行区间。其余约7%是级内与外部间隙的合计，尚不能进一步精确归因为CPU调度、设备调度或IPC等待；本轮也没有做NPUGraph开关对照，不能声称已测得NPUGraph消除了多少百分比的Host开销。即使消除全部这些间隙，采样账本内的理论速度增益也只有约7.6% / 7.4%，不是已验证可实现的提速。不能用profiler设备耗时直接减无profiler TPOT来估算Host预算。

计数器与时间线更支持访存搬运、低batch/小kernel执行效率限制，而非Cube峰值算力或双卡大流量带宽饱和。该判断是本次计数器的综合推断；未测得kernel内有效带宽/峰值FLOPS的完整roofline，不能把全部NPU执行时间直接归为HBM瓶颈。

![双进程与两卡对齐时间线](aligned/r144-dual-process-timeline.png)

'''
text = text.replace('## 时间线账本',intro+'## 时间线账本',1)
l2_section = f'''### L2请求计数与逻辑流量

{traffic}

Memory组的四个`aic/aiv_l2_read/write_bw(GB/s)`字段在两种上下文、两个阶段均返回0，因此本轮直接L2 GB/s不可用，不能将其解释为零访问。上表提供有效的L2命中/缺失次数及独立MemoryAccess逻辑流量；KB按1024字节换算GiB，物理HBM表按十进制GB。两组是不同轮、不同连续位置的采样，不用L2命中率乘GM逻辑流量来反推物理HBM。没有把请求次数擅自换算成物理L2字节量。

'''
text = text.replace('物理HBM与PCIe使用单独的msprof系统采样：',l2_section+'物理HBM与PCIe使用单独的msprof系统采样：',1)
details = '''

PCIe吞吐采样字段在本次短decode窗口中均为0，无法可靠量化稀疏小消息传输；这不表示没有PCIe通信。nonpost_latency的单位是µs，不混入MB/s统计。判断大流量带宽不是首要限制，还依赖已核实的20KiB阶段载荷及拷贝/交接时间线，不能单凭零计数作结论。

本次timing记录每token稳定包含Stage0 1864、Stage1 1885，共3749个kernel。全部设备Memcpy区间并集约0.0215 ms/token，包含embedding、隐状态和标量等拷贝，不能全部称为IPC。阶段间空档约0.72 / 0.74 ms，固定同步/进程交接比纯拷贝执行时间更值得关注；这个空档也不是已分离出的纯IPC成本。

工具停止profiler时输出了RECORD状态警告；已按CPU token边界检查两卡全部目标周期的kernel数量/签名一致，20组阶段×指标检查通过，HBM采样时间窗覆盖超过99%。未使用不完整的Torch→ACL flow映射来补造设备事件。
'''
text = text.replace('\n## 采样与恢复',details+'\n## 采样与恢复',1)
text += '''

恢复后另做一次fresh 8k正式HTTP检查：128输出token，完整回答与r144参考逐字节相同；TTFT 3.935 s，TPOT 33.88 ms。这是恢复验收单次测量，不替代r144三次中位数。记录见post-restore-http.json。

## 数据复核与字段依据

完整Chrome Trace、原始kernel/task CSV压缩包及全时窗HBM/PCIe计数器随证据包交付。`aligned/*_system_counters_full.csv`保留生成时间窗两侧采样点；按前值保持积分，与analyze.py一致。`system-summaries/`保留原工具汇总，但不用于代替decode有效窗的积分结果。SHA256SUMS列出交付文件哈希。

CPU分析复现：在含原始traces/和system/的服务器目录依次运行`python analyze.py`、`python audit.py`、`python finalize.py`；图形使用隔离plot-venv执行`python plot.py`。操作性脚本按实际采集留档，原始私密ENV已经移除；不把历史controller直接当无人值守复跑入口。完整原始文件留在上文服务器目录。

字段口径参见[CANN 9.1 AI Core指令占比、Memory、L2Cache与MemoryAccess定义](https://www.hiascend.com/doc_center/source/zh/CANNCommunityEdition/910/devaids/Profiling/atlasprofiling_16_0069.html)及[torch_npu AiCMetrics API](https://www.hiascend.com/document/detail/en/Pytorch/2610/apiref/customapi/docs/en/custom_APIs/torch_npu-profiler/torch_npu-profiler-AiCMetrics.md)。具体支持度以本次导出和上述无效字段说明为准。
'''
(P/'report.md').write_text(text)
print('FINALIZED_TRAFFIC_UNITS_FULL_COUNTERS_AND_REPORT')
