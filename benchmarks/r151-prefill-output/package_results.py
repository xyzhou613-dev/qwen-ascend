"""Publish verified r151 runtime sources and allowlisted experiment evidence."""
from pathlib import Path
import difflib, hashlib, json, shutil, subprocess, tarfile, urllib.request

P = Path('/models/fnx-diagnostics/r151-prefill-post-20261112')
R = Path('/root/qwen-ascend')
LIVE = Path('/root/ascend-dev/fnx/fnx')
B = P / 'public-evidence'
r = json.load(open(P / 'ab-summary.json'))
pv = json.load(open(P / 'production-verification.json'))
assert r['accepted'] and pv['accepted']
assert not any((P / n).exists() for n in ('lease.active', 'deployment.active', 'original-private.json'))
health = json.load(urllib.request.urlopen('http://127.0.0.1:9002/health', timeout=5))
assert health['service_revision'] == 'r151' and health['prefill_output'] and health['worker_alive']
assert json.load(open(P / 'final-operations.json'))['all_passed']
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=R).strip()
manifest = json.load(open(P / 'deployment-backup/manifest.json'))
for row in manifest['entries']:
    assert hashlib.sha256(Path(row['target']).read_bytes()).hexdigest() == row['new_sha']
(P / 'final-health.json').write_text(json.dumps(health, indent=2))

# Runtime source matches the tested deployment. ELF is delivered as an artifact;
# repository stores source/build instructions, following the existing kernels.
for name in ('engine.py', 'prefill_output.py', 'server.py'):
    shutil.copyfile(LIVE / name, R / 'fnx' / name)
K = R / 'kernels/prefill_output'
K.mkdir(exist_ok=True)
shutil.copyfile(P / 'prefill_output.cpp', K / 'prefill_output.cpp')
build = (R / 'kernels/l2norm_post/build.sh').read_text().replace('norm_post', 'prefill_output')
(K / 'build.sh').write_text(build)
(K / 'build.sh').chmod(0o755)
enable = '''"""Enable the validated prefill output kernel in an r150-compatible launcher."""
import argparse, os
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('launcher',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
raw=a.launcher.read_bytes()
default=b'FNX_PREFILL_OUTPUT=${FNX_PREFILL_OUTPUT:-1}\\n'
passed=b'FNX_PREFILL_OUTPUT=$FNX_PREFILL_OUTPUT '
if default in raw:
 assert raw.count(default)==1 and raw.count(passed)==1,'incomplete or duplicate configuration'
 print('ALREADY_ENABLED');raise SystemExit(0)
anchor=b'FNX_NORM_POST=${FNX_NORM_POST:-1}\\n';assert raw.count(anchor)==1,'unrecognized defaults'
raw=raw.replace(anchor,anchor+default)
anchor=b'FNX_NORM_POST=$FNX_NORM_POST ';assert raw.count(anchor)==1,'unrecognized environment'
raw=raw.replace(anchor,anchor+passed)
if not a.check:
 tmp=a.launcher.with_name(a.launcher.name+'.r151-next');tmp.write_bytes(raw)
 tmp.chmod(a.launcher.stat().st_mode & 0o777);os.replace(tmp,a.launcher)
print('CHECK_OK' if a.check else 'ENABLED')
'''
(R / 'scripts/r151-enable.py').write_text(enable)

b, c = r['baseline'], r['candidate']
prod = json.load(open(P / 'production/summary.json'))
qb = json.load(open(P / 'baseline-quality.json'))
qc = json.load(open(P / 'candidate-quality.json'))
qrows=''
for name,x in qb.items():
 y=qc[name];assert x['nll_tokens']==y['nll_tokens'] and x['argmax']==y['argmax']
 qrows+=f"| {name} | {x['tokens']} | {x['scored']} | {x['mean_nll']:.6f} | {x['top1']:.4f} | 逐位一致 |\n"
table='| 指标 | 同轮r150 | 隔离候选 | 正式r151 | 正式变化 |\n|---|---:|---:|---:|---:|\n'
for label,key,kind in [('8k TTFT中位（s）','median_8k_ttft_s','ttft'),('256k两次TTFT中位（s）','ttft_256k_s','ttft'),('8k TPOT中位（ms）','median_8k_tpot_ms','tpot'),('256k两次TPOT中位（ms）','tpot_256k_ms','tpot'),('缓存256k TPOT（ms、非流式）','cached_256k_tpot_ms','tpot')]:
 change=100*(1-prod[key]/b[key]) if kind=='ttft' else 100*(b[key]/prod[key]-1)
 table+=f"| {label} | {b[key]:.3f} | {c[key]:.3f} | {prod[key]:.3f} | {change:+.2f}% |\n"
components=[json.load(open(P/f'layout-device{d}.json'))['output_2048_assembly']['median_us'] for d in (0,1)]
report=f'''# r151 DeltaNet prefill输出融合：正式验收

2026-11-12，192.168.18.206，2×Ascend 910B4-1，物理3/5、逻辑0/1，24+24两进程流水线。CANN9.1.0、驱动26.1.1、torch2.10.0+cpu、torch_npu2.10.0.post4。起点b356ae8 / r150，仓库干净。

正式启用 `FNX_PREFILL_OUTPUT=1`，健康版本r151，保留r150的 `FNX_NORM_ORDERED=1`。新内核只融合DeltaNet prefill的输出相加、BF16转换和布局写入；FP32递推、BMM运算及其顺序保持原路径。隔离A/B通过后，经原p275.sh正式入口重测。

## 整模型HTTP实测

{table}

TTFT变化按`1-新TTFT/基线TTFT`计算，TPOT列变化表示输出速度`基线TPOT/新TPOT-1`。8k为8064输入+128输出，每版本三份固定输入取中位。长文为262016输入+128输出，每版本两份不同文档标识的固定输入，两次均为fresh并核实新增prefill；两次中位即两值平均，不外推长输入追加8k输出。缓存重复为第一次长请求的非流式命中，新增prefill=0，单列。

基线两次长文TTFT（s）：{b['ttft_256k_runs_s']}；隔离候选：{c['ttft_256k_runs_s']}；正式入口：{prod['ttft_256k_runs_s']}。正式逐输入TTFT降低（%）：{pv['individual_long_reductions_pct']}。门禁要求中位降低>1%、每次降低>0.5%，同时8k TTFT及两场景TPOT不能回退>1%；精度通过后才判断性能。

原FNX、manager与已识别watchdog在受控维护期间停止，基线与候选分别冷启动，只在127.0.0.1:19002测量，每次仅一个完整Engine。冷启动/图捕获不算HTTP TTFT。 本轮隔离启动到HTTP ready为 {json.load(open(P/'baseline-startup.json'))['start_to_http_ready_s']:.1f} / {json.load(open(P/'candidate-startup.json'))['start_to_http_ready_s']:.1f}s（基线/候选），候选启动更慢；未单独拆分原因，不计作请求延迟。独立恢复任务保护断链与超时；隔离结束先恢复r150，再安装正式五个受控文件，9002上重新测量。没有将组件加速比例当作整模型加速比例。

## 实施与数值边界

原路径在每个64-token块将两个BMM结果以FP32相加，保留全部FP32块，最后transpose/cat并转BF16。新路径预分配token-major BF16输出，逐块读取同样的两个FP32 BMM结果，执行FP32 Add，再按RINT转换BF16，直接写入该块目标位置，去掉末尾cat/cast和FP32输出中间存储。转换移到每块相加之后，与原最终转换逐元素同序；没有FMA或新的Exp/Rsqrt，也没有改变矩阵乘法或递推结合顺序。

固定输入为两个连续FP32 `[48,64,128]`，输出为连续BF16 `[块数*64,48,128]`，生产默认mode4，每任务gather8个head并连续写目标。形状/dtype/连续性不符时使用原相加、转置和转换的拷贝兜底；`FNX_PREFILL_OUTPUT=0`恢复原函数路径。小于等于64的分支及decode不走新内核。

满2048-token窗口中，仅这条输出组装链的逻辑读写量约312MiB→120MiB，节省192MiB；这不是实测物理HBM字节，L2缓存会影响外存流量。两卡32块完整输出组装图事件中位分别 {components[0]['torch']:.1f}→{components[0]['direct4']:.1f}µs、{components[1]['torch']:.1f}→{components[1]['direct4']:.1f}µs，测的是整条组装链，不是整个DeltaNet函数。

另试过融合FP32状态 `current*decay+delta`，两卡8次更新的原/候选图事件均约290µs，没有稳定收益，未进入正式engine。只合并输出Add/Cast但保留最终cat的方案约745→640–676µs，也未作为最终实现。原型源码与组件数据保留，避免把未采用的状态融合算入交付。

## 精确性与范围

双卡各64组不同幅值、相消和舍入边界输入，5种直接输出布局逐位通过，目标块前后区域不变；2048-token完整拼装也逐位一致。全部36个真实权重DeltaNet层覆盖65/128/512 token eager及4次128-token动态graph replay，每层7次，共252例；输出、BF16卷积历史与FP32状态逐位相同。四个完整层L00/L12/L24/L44的decode及128/512/2048 prefill也通过。

| 冻结文本 | token数 | 目标数 | 两版本NLL | top1 | 对照 |
|---|---:|---:|---:|---:|---|
{qrows}

五份共995个目标用真实plan_prefill/prefill_chunk/finish_prefill评分，逐目标NLL和argmax全部相同。语料与r150冻结集合相同，新增自然文本的URL和哈希见quality-input-manifest.json；12k仍是重复服务文本，不能单独代表长文质量，历史100题未重跑。三份fresh 8k、两份fresh 256k和缓存完整回答，在隔离及正式入口均与r150基线相同；两份长请求均三事实召回3/3。样本通过不证明所有输入等价。

## 部署与复现

正式修改engine.py、server.py、p275.sh，新增prefill_output.py和libfnx_prefill_output.so；其余r150文件哈希不变。仓库 `kernels/prefill_output/` 保存C++与Bisheng构建脚本，Linux aarch64 ELF在证据implementation/。启动补丁 `python3 scripts/r151-enable.py /root/ascend-dev/p275.sh --check` 可检查，去掉--check才写入。复用同一模型配置、BF16激活/KV及FP32递推状态。

部署按文件哈希保护并保存精确备份；失败则恢复五个旧文件并通过原入口启动r150。正式server不包含质量诊断路由。最终manager active、RefuseManualStart=no，一份正常PPID1 watchdog恢复；临时timer、drop-in、租约、私有完整ENV快照与诊断进程已清理。证据仅白名单打包，不含权重、图缓存或用户请求激活。原始目录 `/models/fnx-diagnostics/r151-prefill-post-20261112`。
'''
(P/'report.md').write_text(report)
block=f'''<!-- r151-current -->
## 2026-11-12 r151 正式验收：DeltaNet prefill输出融合

当前生产r151，`FNX_PREFILL_OUTPUT=1`，保留r150归一化融合。仅将64-token块输出的FP32相加、BF16转换、布局写入合并，直接填充最终输出缓冲；BMM与FP32状态递推不变。未采用无稳定收益的状态Mul/Add融合。

{table}

8k三次中位；长文为262016输入+128输出，每版本两份不同标识的fresh请求。五份文本995个目标NLL/argmax、全部36个真实DeltaNet层、四个完整层逐位检查通过；完整回答和缓存一致。原入口、manager、正常watchdog恢复，临时恢复任务和私有ENV已清理。详见[r151验收](r151-prefill-output.md)和benchmarks/r151-prefill-output/。以下为历史记录。
<!-- /r151-current -->
'''
(P/'techdoc-block.md').write_text(block)
doc=R/'docs/Qwen3.8-Flash-Next-2x910B4-技术文档.md';text=doc.read_text();assert '<!-- r151-current -->' not in text
head,rest=text.split('\n',1);doc.write_text(head+'\n\n'+block+'\n'+rest.lstrip('\n'))
(R/'docs/r151-prefill-output.md').write_text(report+'\n[完整证据](../benchmarks/r151-prefill-output/README.md)。\n')
B.mkdir(exist_ok=True)
names = ['report.md', 'ab-summary.json', 'production-verification.json', 'final-operations.json', 'baseline-reference-check.json',  'quality-input-manifest.json', 'input-manifest.json', 'candidate-source-hashes.json', 'production-before-hashes.json', 'production-selected-config.json', 'baseline-quality.json', 'candidate-quality.json', 'baseline-health.json', 'candidate-health.json', 'baseline-startup.json', 'candidate-startup.json', 'component-device0.json', 'component-device1.json', 'all-linear-device0.json', 'all-linear-device1.json', 'layer-probe.json', 'final-health.json', 'production-health.json', 'fixtures.json', 'engine.patch',  'http_bench.py', 'quality_client.py', 'evaluate.py', 'component_probe.py',  'layer_probe.py',  'prefill_nll.py', 'launch.py', 'snapshot.py', 'stop_owned.py', 'controller.sh', 'rescue.sh', 'deployment_files.py', 'deploy_stop.py', 'deploy.sh', 'deploy_rescue.sh', 'package_results.py']
names += ['release-checks.json','baseline-config-check.json','candidate-config-check.json','check_config.py','final_ops.py','layout-device0.json','layout-device1.json','long-repeat-manifest.json','layout_probe.py','freeze_long_repeat.py','component_launch.py','layout_launch.py','all_prefill_probe.py','all_prefill_launch.py','prefill_post.cpp','build_output.sh']
for name in names:
    shutil.copyfile(P / name, B / name)
for folder in ('baseline', 'candidate', 'production', 'inputs'):
    dest = B / folder; dest.mkdir(exist_ok=True)
    for f in (P / folder).glob('*.json'): shutil.copyfile(f, dest / f.name)
dest = B / 'implementation'; dest.mkdir(exist_ok=True)
for name in ('prefill_output.py', 'libfnx_prefill_output.so'): shutil.copyfile(LIVE / name, dest / name)
for name in ('prefill_output.cpp', 'build.sh'): shutil.copyfile(K / name, dest / name)
shutil.copyfile(R / 'scripts/r151-enable.py', dest / 'r151-enable.py')
shutil.copyfile(P / 'deployment-backup/manifest.json', B / 'deployment-manifest.json')
(B / 'README.md').write_text("""# r151 DeltaNet prefill output assembly

Accepted only after exactness and end-to-end TTFT gates. Read report.md, ab-summary.json and production-verification.json. Three fresh 8k requests, two distinct fresh 262016+128 long requests and a nonstream cached repeat per version. Five frozen documents/995 targets use the real prefill path. All 36 real-weight DeltaNet layer subpaths and four complete layers were checked, including FP32 state bits.

Implementation uses FP32 addition followed by BF16 RINT conversion and writes token-major output directly. BMM and FP32 recurrence stay unchanged. The default kernel gathers eight heads per task, fixed [48,64,128] inputs; unsupported shapes fall back. Five benchmark layouts are included in the build, while production selects mode4. Linux aarch64 ELF must be paired with this source/CANN build. See implementation/ and kernels/prefill_output/.

Initial state Mul/Add fusion and output Add/Cast with a final cat were not selected. Their standalone prototype source, wrapper and library are kept in rejected-state-and-cat/. This does not enable those paths in production.

Operational scripts are historical and require the fixed server paths and a maintenance window. Full process ENV, credentials, model weights and graph caches are excluded. SHA256SUMS covers distributed files.
""")
rejected=B/'rejected-state-and-cat';rejected.mkdir(exist_ok=True)
for n in ('prefill_post.py','libfnx_prefill_post.so'):shutil.copyfile(P/'fnx'/n,rejected/n)
for n in ('prefill_post.cpp','build.sh'):shutil.copyfile(P/n,rejected/n)
files = sorted(f for f in B.rglob('*') if f.is_file() and f.name != 'SHA256SUMS')
(B / 'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + str(f.relative_to(B)) + '\n' for f in files))
with tarfile.open(P / 'r151-prefill-output-evidence.tgz', 'w:gz') as tar:
    tar.add(B, arcname='r151-prefill-output-evidence')
shutil.copytree(B, R / 'benchmarks/r151-prefill-output')
print('VERIFIED_R151_RESULTS_PACKAGED',len(files),flush=True)
