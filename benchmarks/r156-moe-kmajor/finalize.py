import difflib,hashlib,json,pathlib,shutil,statistics,subprocess,tarfile,time,urllib.request
P=pathlib.Path(__file__).resolve().parent;R=pathlib.Path('/root/qwen-ascend')
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def read(name):return json.load(open(P/name))
def command(*args):return subprocess.check_output(args,text=True).strip()
h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5));assert h['service_revision']=='r152' and h['worker_alive'] and h['ctx']==262144
before=read('production-before-hashes.json');live=pathlib.Path('/root/ascend-dev/fnx/fnx')
assert all(sha(live/n)==s for n,s in before.items())
assert sha(pathlib.Path('/root/ascend-dev/p275.sh'))==(P/'p275-before.sha256').read_text()
assert command('systemctl','is-active','model-switch.service')=='active'
assert command('systemctl','show','model-switch.service','--value','-p','RefuseManualStart')=='no'
assert not command('systemctl','show','model-switch.service','--value','-p','DropInPaths')
assert read('recovered-http.json')['answer_matches_preboot_r152']
procs=[];watchdogs=[]
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  cmd=[s.decode(errors='replace') for s in proc.joinpath('cmdline').read_bytes().split(b'\0') if s]
  if 'fnx.server' in cmd or 'fnx.pipeline_worker' in cmd:
   assert str(proc.joinpath('cwd').resolve())=='/root/ascend-dev/fnx';procs.append(dict(pid=int(proc.name),module='fnx.server' if 'fnx.server' in cmd else 'fnx.pipeline_worker'))
  if len(cmd)>1 and cmd[0].endswith('bash') and cmd[1]=='/root/ascend-dev/watchdog.sh':watchdogs.append(int(proc.name))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
assert sorted(r['module'] for r in procs)==['fnx.pipeline_worker','fnx.server'] and len(watchdogs)==1
units={u:command('systemctl','show',u,'--value','-p','ActiveState') for u in ['fnx-r156-recover','fnx-r156-recover-v2','fnx-r156-quality-r152','fnx-r156-layerpair-v2','fnx-r156-merge']}
assert all(v not in ('active','activating','reloading') for v in units.values())
devices=json.loads(command('docker','inspect','-f','{{json .HostConfig.Devices}}','bole-model-Flash'))
assert {r['PathOnHost'] for r in devices if r['PathOnHost'].split('/')[-1] in ['davinci0','davinci1','davinci3','davinci5']}=={'/dev/davinci0','/dev/davinci1'}
ops=dict(timestamp_epoch=time.time(),health=h,production_files_unchanged=True,source_file_count=len(before),p275_unchanged=True,
 production_processes=procs,watchdogs=watchdogs,manager_active=True,manager_refuse_manual_start=False,manager_temporary_dropins=False,
 experimental_units=units,container_devices=devices,container_preserved=True,candidate_deployed=False,
 host_boot_id=pathlib.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),host_reboot_cause='unconfirmed',
 recovered_http_verified=True,private_recovery_backup='server-only, mode 0700; excluded from artifacts')
(P/'final-operations.json').write_text(json.dumps(ops,indent=2))
layers=[r for d in (0,1) for r in read(f'layers-device{d}.json')['records']]
means={kind:{mode:statistics.mean(r['median_ms'][mode] for r in layers if r['kind']==kind) for mode in ('w8a8','combined')} for kind in ('linear','full')}
proxy=sum((means[k]['w8a8']-means[k]['combined'])*count for k,count in [('linear',36),('full',12)])
merge={mode:statistics.mean(read(f'merge-device{d}.json')['median_us'][mode] for d in (0,1)) for mode in ('baseline','split','fused')}
summary=dict(status='component_candidate_not_deployed',layer_means_ms=means,weighted_48_layer_saving_proxy_ms=proxy,
 production_down_epilogue_us=merge,full_model_tpot_measured=False,quality_acceptance_complete=False,
 quality_inputs=read('quality-freeze.json'),nll_scored_targets_frozen=read('nll-freeze.json')['scored_targets'],
 r152_screen=read('quality-r152screen-summary.json'),service_restored=True,host_reboot_changed_device_indices=True)
(P/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
screen=summary['r152_screen']
report=f'''# r156：原布局MoE down、独立质量输入与重启恢复

2026-11-13，起点291d5c7。正式服务仍r152；本轮候选未发布，没有新的整模型TPOT结果。

## 已实施与真实硬件结果

新增AscendC down kernel，直接读生产expert-major INT4 [512,640,2560/8]及BF16 [512,10,2560] scale，TOPK10动态专家，BF16激活/输出。按64个K值分组，使用Brcb广播及FP32分组规约；不生成选中专家GM权重，不做权重转置，不保留双份down权重。输出merge复用原epilogue的BF16乘法舍入、顺序求和、gate与shared路径，只将输入改为10个对角结果。默认开关关闭。

早期93µs对照是apply_diag，含独立对角提取；生产FNX_MOE_EPILOGUE=1已融合这步，不能据此声称生产down节省50%。补齐真实融合入口后，动态专家工作集、两卡交替图计时的正确对照为：

| 完整down+epilogue，两卡中位数均值 | µs |
|---|---:|
| 生产gather+W4A16+融合epilogue | {merge['baseline']:.3f} |
| 原布局down + 对角epilogue，两kernel，选定 | {merge['split']:.3f} |
| down及epilogue合成一个kernel，否决 | {merge['fused']:.3f} |

单kernel版虽然与两kernel输出逐位一致，但执行更慢，保留失败源码与结果，不默认启用。并非融合越多越快。

完整FnLayer对照为SmoothQuant W8A8 / W8A8+新down，真实L00/L24 linear及L03/L27 full，位置4096、私有窗口8192，16次forward/图、每轮2次replay、12轮交替顺序。两卡均值：

| 类型 | W8A8 ms | 组合 ms |
|---|---:|---:|
| linear | {means['linear']['w8a8']:.6f} | {means['linear']['combined']:.6f} |
| full | {means['full']['w8a8']:.6f} | {means['full']['combined']:.6f} |

按36+12层加权的额外节省代理为{proxy:.6f}ms/token。仅4层、合成hidden和重复图；不能直接从r155的29.38ms减去它宣称已取得新TPOT，更不能与较早93µs组件参考相加。组件和整层的权重工作集不同，收益比例也不同。

动态专家覆盖全部512，含重复专家511、四档输入幅值。down对生产输出最大relative L2约0.00233，独立CPU FP32公式误差约0.00165；不是任务准确率下降比例。四个完整层输出最大relative L2约0.00243，conv及测试中的FP32状态精确一致。每层33次实际路径命中审计；新输入NPUGraph/eager检查通过。其余层间误差累积、真实prefill及任务质量尚未验收。探针仅加载一个权重对象或共享权重的层对，每进程allocator上限1/32，保留服务健康及显存余量检查，不构建第二个完整Engine。

第一次整层接入挂在apply_diag，实际被生产融合分支绕过，已捕获并修正，bypassed记录保留。另有多编译单元g_tilingKey链接冲突，合为同一编译单元后解决，失败日志保留。

## 独立质量输入与评测框架

冻结1100题：CMMLU 300（67科）、MMLU-Pro 300（此次来源池覆盖12类）、GSM8K 200、HumanEval 100、MBPP官方test范围100、合成JSON提取/参数100。后一项不代表原生tool-call协议验收。来源按官方revision与文件SHA固定；与FNX校准/开发数据分离，但不能声称这些公共题目从未进入原模型预训练。

另用实际部署tokenizer冻结196段、100352个计分目标，中英知识/数学/代码各25088；与r154校准/开发无共同连续128-token窗口。它不是自然文本和多长度长文prefill验收，也尚未计算候选NLL。原始题目、答案、测试及token IDs保存在server-only private-quality，公开制备脚本、源manifest和hash，不重分发原始数据副本。

评测器支持断点续跑、固定输入SHA、MCQ/数值/严格JSON判定；代码在无网络、只读、非root、限CPU/内存/PID的临时容器里执行，拒绝不通过的代码自测，未把生成代码放到模型容器或宿主环境执行。框架自测通过；在线r152仅做32题MCQ/JSON试跑，{screen['correct']}/{screen['completed']}，这是小批基线检查，不是1100题准确率，也没有候选质量结论。100k NLL与1100任务的完整配对仍待运行，既定质量门槛不变。

## 本轮发生整机重启，正式服务已恢复

约17:02主机重启，原因未确认，未执行reboot操作。当前journal只保留新boot，pstore无文件，没有据此判定断电或kernel故障。断链发生在已完成组件结果后、数据上传期间；不能据时间先后推断因果。

重启后相同PCI地址的两张卡枚举由3/5变成0/1，原容器因/dev/davinci3不存在启动失败。用户明确确认无维护并要求恢复服务。确认所有容器停止、NPU无任务后，备份原HostConfig，停Docker daemon，将bole-model-Flash两个设备节点映射修正为当前0/1，再启动daemon及原容器。保留原container ID与可写层，没有删除/重建容器、没有改模型源码或p275。原设备配置备份留在0700私有恢复目录，不打入证据包。

通过原p275.sh恢复r152、manager及单份正常watchdog。9002 fresh 8k/128输出烟测回答与重启前r152完全相同；33.18ms只是一轮恢复烟测，不是新硬件状态的优化基线。45个部署源码/ELF及p275 SHA均保持一致。临时实验unit均已结束，无临时RefuseManualStart/drop-in。硬件枚举改变后，下一轮必须重新建立同配置整模型基线；本轮停止继续停服实验。

## 下一项验收

先将已冻结质量输入完整配对运行，补齐自然文本与真实多长度prefill；在受控窗口重建新枚举下的r152/W8A8/组合对照与新PMU账本。组合候选只有通过质量、完整层生命周期和HTTP A/B/A后才能讨论发布。本轮不宣称达到累计20%，不发布未验收的数值路径。

诊断目录/models/fnx-diagnostics/r156-moe-kmajor-20261113。公开源码及指标在experiments/r156-moe-kmajor与benchmarks/r156-moe-kmajor，不含模型权重、校准向量、完整进程ENV、Docker原始配置或凭据。

API依据：[Brcb及stride语义](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0089.html)。数据依据：[CMMLU](https://github.com/haonan-li/CMMLU)、[MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)、[GSM8K](https://github.com/openai/grade-school-math)、[HumanEval](https://github.com/openai/human-eval)、[MBPP](https://github.com/google-research/google-research/tree/master/mbpp)。
'''
(P/'report.md').write_text(report)
block=f'''## r156 原布局down与质量输入（2026-11-13，生产仍r152）

真实生产down+epilogue组件约{merge['baseline']:.2f}→{merge['split']:.2f}µs；四个完整层在W8A8上再省约{proxy:.2f}ms/token的48层加权代理，尚无新HTTP TPOT。单kernel融合更慢，否决。冻结1100题及100352个NLL目标，框架自测和32题r152试跑完成，完整质量配对未运行。

17:02整机重启后设备枚举变为0/1，原容器旧3/5节点失效。经用户确认恢复，保留容器并修正设备映射，原p275恢复9002、manager和watchdog；45个部署文件及启动脚本未变，8k回答与重启前r152一致。重启原因未知，下一轮需重建性能基线。详见[r156记录](r156-moe-kmajor.md)。

'''
(P/'techdoc-block.md').write_text(block)
print('FINALIZED',proxy,merge,'service healthy, source unchanged; no end-to-end candidate claim')
