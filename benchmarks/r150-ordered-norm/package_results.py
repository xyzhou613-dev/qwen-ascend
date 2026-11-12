"""Publish verified r150 runtime sources and allowlisted experiment evidence."""
from pathlib import Path
import difflib, hashlib, json, shutil, subprocess, tarfile, urllib.request

P = Path('/models/fnx-diagnostics/r150-ordered-norm-20261112')
R = Path('/root/qwen-ascend')
LIVE = Path('/root/ascend-dev/fnx/fnx')
B = P / 'public-evidence'
r = json.load(open(P / 'ab-summary.json'))
pv = json.load(open(P / 'production-verification.json'))
assert r['accepted'] and pv['accepted']
assert not any((P / n).exists() for n in ('lease.active', 'deployment.active', 'original-private.json'))
health = json.load(urllib.request.urlopen('http://127.0.0.1:9002/health', timeout=5))
assert health['service_revision'] == 'r150' and health['norm_ordered'] and health['worker_alive']
assert json.load(open(P / 'final-operations.json'))['all_passed']
assert not subprocess.check_output(['git', 'status', '--porcelain'], cwd=R).strip()
manifest = json.load(open(P / 'deployment-backup/manifest.json'))
for row in manifest['entries']:
    assert hashlib.sha256(Path(row['target']).read_bytes()).hexdigest() == row['new_sha']
(P / 'final-health.json').write_text(json.dumps(health, indent=2))

# Runtime source matches the tested deployment. ELF is delivered as an artifact;
# repository stores source/build instructions, following the existing kernels.
for name in ('engine.py', 'norm_ordered.py', 'server.py'):
    shutil.copyfile(LIVE / name, R / 'fnx' / name)
K = R / 'kernels/l2norm_ordered'
K.mkdir(exist_ok=True)
shutil.copyfile(P / 'norm_ordered.cpp', K / 'norm_ordered.cpp')
build = (R / 'kernels/l2norm_post/build.sh').read_text().replace('norm_post', 'norm_ordered')
(K / 'build.sh').write_text(build)
(K / 'build.sh').chmod(0o755)
enable = '''"""Enable the validated fixed-shape q/k kernel in an r144-compatible launcher."""
import argparse, os
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('launcher',type=Path);p.add_argument('--check',action='store_true');a=p.parse_args()
raw=a.launcher.read_bytes()
default=b'FNX_NORM_ORDERED=${FNX_NORM_ORDERED:-1}\\n'
passed=b'FNX_NORM_ORDERED=$FNX_NORM_ORDERED '
if default in raw:
 assert raw.count(default)==1 and raw.count(passed)==1,'incomplete or duplicate configuration'
 print('ALREADY_ENABLED');raise SystemExit(0)
anchor=b'FNX_NORM_POST=${FNX_NORM_POST:-1}\\n';assert raw.count(anchor)==1,'unrecognized defaults'
raw=raw.replace(anchor,anchor+default)
anchor=b'FNX_NORM_POST=$FNX_NORM_POST ';assert raw.count(anchor)==1,'unrecognized environment'
raw=raw.replace(anchor,anchor+passed)
if not a.check:
 tmp=a.launcher.with_name(a.launcher.name+'.r150-next');tmp.write_bytes(raw)
 tmp.chmod(a.launcher.stat().st_mode & 0o777);os.replace(tmp,a.launcher)
print('CHECK_OK' if a.check else 'ENABLED')
'''
(R / 'scripts/r150-enable.py').write_text(enable)

b, c = r['baseline'], r['candidate']
prod = json.load(open(P / 'production/summary.json'))
qb = json.load(open(P / 'baseline-quality.json'))
qc = json.load(open(P / 'candidate-quality.json'))
qrows = ''
for name, x in qb.items():
    y = qc[name]
    assert x['nll_tokens'] == y['nll_tokens'] and x['argmax'] == y['argmax']
    qrows += f"| {name} | {x['tokens']} | {x['scored']} | {x['mean_nll']:.6f} | {x['top1']:.4f} | 全部逐位相同 |\n"
table = '| 场景 | 同轮r144 | 隔离候选 | 正式r150 | 正式输出速度变化 |\n|---|---:|---:|---:|---:|\n'
for name, key in [('fresh 8k（三次中位）', 'median_8k_tpot_ms'), ('fresh 256k（各一次）', 'tpot_256k_ms'), ('缓存256k（各一次、非流式）', 'cached_256k_tpot_ms')]:
    table += f"| {name} | {b[key]:.2f} | {c[key]:.2f} | {prod[key]:.2f} | {(b[key]/prod[key]-1)*100:+.2f}% |\n"

report = f'''# r150 固定归约顺序的 q/k 全融合：正式部署与复测

2026-11-12，192.168.18.206，2×Ascend 910B4-1（物理3/5，逻辑0/1，24+24两进程），CANN9.1.0、驱动26.1.1、torch2.10.0+cpu、torch_npu2.10.0.post4。本轮起点为r144实现794a22e，仓库HEAD cacad4d且干净。

正式启用 `FNX_NORM_ORDERED=1`，由原p275.sh入口加载，健康版本r150。两个场景均通过精度与性能门禁，并在正式入口再次复现收益。此前r148候选因不精确被否决；r149保留CANN归约虽精度通过，但8k仅+0.119%、256k反降0.174%，也未发布。

## 实际TPOT（ms/token）

{table}

输出速度变化为`TPOT基线/TPOT新版本-1`。8k输入8064、输出128，每版本三份固定输入取TPOT中位；256k输入262016、输出128，每版本一次，不能外推长输入追加8k输出。缓存重复新增prefill=0，且使用非流式接口，单列展示，不冒充fresh收益。

HTTP TTFT：8k中位 {b['median_8k_ttft_s']:.3f}→{prod['median_8k_ttft_s']:.3f}s，256k {b['ttft_256k_s']:.3f}→{prod['ttft_256k_s']:.3f}s。该改动针对decode，**没有确认长文TTFT改善**，不能把单次微小差异解释为prefill提速，也未实现最初20%的输出加速目标。

基线、候选各冷启动，隔离在127.0.0.1:19002；原FNX、watchdog和管理器在受控维护流程内停止，每次只有一个Engine，使用同配置、权重和冻结输入。独立限时恢复任务保护断链场景。HTTP ready前的图捕获/启动时间不混入TTFT。隔离A/B后先恢复r144，再安装仅五个受控文件并通过正式入口重测。

## 为什么这一版能修复旧反例

原r144是两次BF16平方、CANN ReduceSum、查表后处理，加query缩放，每个线性层7个kernel。新路径在固定单token `[1,10240]` 卷积/SiLU输出上用1个kernel同时处理q/k，36层减少约216个kernel/token。BF16平方舍入、归约到BF16、原CANN生成的epsilon/Rsqrt查表、归一化BF16舍入和query缩放舍入均保留。

r148的原生128元素归约与CANN累加顺序不同。新版本先将两个64元素片段逐元素相加，再对64元素归约，修复四个已保存的稀疏边界反例。该顺序是在本机CANN9.1、固定R=128形状上对拍得到的，不证明其他CANN版本或任意形状同序。不同batch、dtype、shape或非连续输入回到原路径；`FNX_NORM_ORDERED=0`可停用新分支。没有改FP32递推、MatMul、prefill BMM或模型权重。

## 精确性检查

双卡各4608组随机/SiLU/原始BF16编码和稀疏边界输入，以及四个历史反例均逐位通过。全部36个真实权重DeltaNet子路径，逐层16次eager+8次动态graph replay，共864例，输出、BF16卷积历史与FP32状态逐位相同。完整L00/L12/L24/L44的decode与128/512/2048 prefill也逐位通过。组件图事件中位约16.8→9.1µs，整模型收益以上表为准。

| 冻结文本 | token数 | 目标数 | 两版本平均NLL | top1 | NLL/argmax对照 |
|---|---:|---:|---:|---:|---|
{qrows}

五份共995个尾部目标，使用真实 `plan_prefill/prefill_chunk/finish_prefill`，不使用逐token loss接口代替prefill验证。新增18k/36k为鲁迅自然文本，来源为维基文库《阿Q正传》及《彷徨》篇目，按清单抽取段落，只连接一次并取不重叠token切片，没有重复填充；URL、HTML/文本/ID哈希见quality-input-manifest.json。原12k服务文本仍是重复样本，不能单独代表长文质量。样本通过不构成所有输入等价的证明，历史100题未重跑。

隔离三份fresh 8k完整回答、256k完整回答和缓存回答相同；正式入口再次全部回答精确相同，256k三个事实召回3/3。保留所有逐目标NLL、argmax、HTTP统计与冻结输入。

## 部署、回退与复现

新增 `fnx/norm_ordered.py`、`kernels/l2norm_ordered/`，改engine的单token分支、health版本和启动开关。Linux aarch64 ELF在部署证据implementation/内，仓库保存源码与Bisheng构建脚本：`bash kernels/l2norm_ordered/build.sh /root/ascend-dev/fnx/fnx/libfnx_norm_ordered.so`。复用已有norm_post查表；需部署相同版本的Python/ELF、开启开关后重启以重新捕获图。

启动参数补丁为 `python3 scripts/r150-enable.py /root/ascend-dev/p275.sh --check`，去掉--check才写入；可重复检查。回退优先恢复deployment-backup中五个文件的原内容并由原入口重启，或设置FNX_NORM_ORDERED=0后重新捕获图。不会把诊断质量路由放到正式server.py。

最终健康检查、五个部署文件哈希、正式HTTP复测和管理器/watchdog状态均已验收。临时租约、恢复timer、drop-in和完整ENV快照已移除。原始实验目录 `/models/fnx-diagnostics/r150-ordered-norm-20261112`；证据采用白名单打包，不含完整ENV、权重、图缓存或用户请求激活。
'''
(P / 'report.md').write_text(report)
block = f'''<!-- r150-current -->
## 2026-11-12 r150 正式验收：修复归约顺序并融合q/k

当前生产为r150，`FNX_NORM_ORDERED=1`。固定R=128按两个64元素片段先相加再归约，保留原BF16舍入及CANN查表，36层归一化由每层7个kernel变为1个。r148的不精确全融合与r149无整模型收益的保守融合均未发布。

{table}

8k三次中位、256k各一次；256k为262016输入+128输出。正式256k HTTP TTFT {b['ttft_256k_s']:.3f}→{prod['ttft_256k_s']:.3f}s，没有确认长文TTFT改善。双卡边界探针、全部36个真实DeltaNet子路径、四个完整层逐位通过；五份6k/12k/18k/24k/36k文本共995个目标NLL/argmax全部相同，fresh与缓存完整回答均一致。历史100题未重跑。原入口、manager和正常watchdog已恢复，无临时恢复任务。

详见[r150正式验收](r150-ordered-norm.md)及[r149否决记录](r149-cann-reduce.md)。以下为历史记录。
<!-- /r150-current -->
'''
(P / 'techdoc-block.md').write_text(block)
doc = R / 'docs/Qwen3.8-Flash-Next-2x910B4-技术文档.md'
text = doc.read_text(); assert '<!-- r150-current -->' not in text
head, rest = text.split('\n', 1); doc.write_text(head + '\n\n' + block + '\n' + rest.lstrip('\n'))
(R / 'docs/r150-ordered-norm.md').write_text(report + '\n完整证据见[证据目录](../benchmarks/r150-ordered-norm/README.md)。\n')

B.mkdir(exist_ok=True)
names = ['report.md', 'ab-summary.json', 'production-verification.json', 'final-operations.json', 'baseline-reference-check.json', 'variant-config-check.json', 'quality-input-manifest.json', 'input-manifest.json', 'candidate-source-hashes.json', 'production-before-hashes.json', 'production-selected-config.json', 'baseline-quality.json', 'candidate-quality.json', 'baseline-health.json', 'candidate-health.json', 'baseline-startup.json', 'candidate-startup.json', 'component-device0.json', 'component-device1.json', 'all-linear-device0.json', 'all-linear-device1.json', 'layer-probe.json', 'final-health.json', 'production-health.json', 'fixtures.json', 'engine.patch', 'freeze_natural.py', 'http_bench.py', 'quality_client.py', 'evaluate.py', 'component_probe.py', 'all_linear_probe.py', 'layer_probe.py', 'online_launch.py', 'prefill_nll.py', 'launch.py', 'snapshot.py', 'stop_owned.py', 'controller.sh', 'rescue.sh', 'deployment_files.py', 'deploy_stop.py', 'deploy.sh', 'deploy_rescue.sh', 'package_results.py']
names.append('final_ops.py')
for name in names:
    shutil.copyfile(P / name, B / name)
for folder in ('baseline', 'candidate', 'production', 'inputs'):
    dest = B / folder; dest.mkdir(exist_ok=True)
    for f in (P / folder).glob('*.json'): shutil.copyfile(f, dest / f.name)
dest = B / 'implementation'; dest.mkdir(exist_ok=True)
for name in ('norm_ordered.py', 'libfnx_norm_ordered.so'): shutil.copyfile(LIVE / name, dest / name)
for name in ('norm_ordered.cpp', 'build.sh'): shutil.copyfile(K / name, dest / name)
shutil.copyfile(R / 'scripts/r150-enable.py', dest / 'r150-enable.py')
shutil.copyfile(P / 'deployment-backup/manifest.json', B / 'deployment-manifest.json')
(B / 'README.md').write_text('''# r150 ordered q/k normalization

Accepted and deployed after exactness and speed gates. Read report.md, ab-summary.json and production-verification.json. Five documents/995 scored targets use the actual prefill policy; fixture IDs and source hashes are frozen. Three fresh 8k inputs and one near-capacity 256k input per version, plus a nonstream cached repeat. Production separately reproduces exact answers and speed gains.

Implementation is specialized to BF16 single-token contiguous [1,10240], head dimension128, CANN9.1 on 910B4. Other inputs fall back. Existing norm_post builds the CANN lookup table; do not substitute a generic rsqrt. ELF is Linux aarch64 and paired with the distributed source. Experimental controllers are historical server-specific operational evidence; do not execute without site checks and maintenance authorization. Full process ENV, credentials, model weights and graph caches are excluded. SHA256SUMS covers distributed files.
''')
files = sorted(f for f in B.rglob('*') if f.is_file() and f.name != 'SHA256SUMS')
(B / 'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest() + '  ' + str(f.relative_to(B)) + '\n' for f in files))
with tarfile.open(P / 'r150-ordered-norm-evidence.tgz', 'w:gz') as tar:
    tar.add(B, arcname='r150-ordered-norm-evidence')
shutil.copytree(B, R / 'benchmarks/r150-ordered-norm')
old = Path('/models/fnx-diagnostics/r149-cann-reduce-20261112')
shutil.copytree(old / 'public-evidence', R / 'benchmarks/r149-cann-reduce')
(R / 'benchmarks/r149-cann-reduce/README.md').write_text('Read report.md and outcome.json. Candidate rejected: precision passed, no reproducible whole-model gain. Runtime not deployed. Source and diagnostics are historical evidence.\n')
oldb=R/'benchmarks/r149-cann-reduce'
(oldb/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(oldb))+'\n' for f in sorted(oldb.rglob('*')) if f.is_file() and f.name!='SHA256SUMS'))
(R / 'docs/r149-cann-reduce.md').write_text((old / 'report.md').read_text())
print('VERIFIED_R150_RESULTS_PACKAGED', len(files), flush=True)
