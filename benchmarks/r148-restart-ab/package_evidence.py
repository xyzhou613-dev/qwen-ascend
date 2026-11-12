"""Package the measured but rejected candidate; production files stay r144."""
import difflib,hashlib,json,shutil,statistics,subprocess,tarfile,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;R=Path('/root/qwen-ascend');C=Path('/models/fnx-diagnostics/r146-norm-pair-20261112')
B=P/'public-evidence';B.mkdir(exist_ok=True)
r=json.load(open(P/'ab-summary.json'));assert not r['accepted']
assert not (P/'lease.active').exists() and not (P/'original-private.json').exists()
post=json.load(open(P/'post-restore-http.json'));assert post['answer_exact'] and post['stats']['n_prefill']>=8060
before=json.load(open(P/'production-before-hashes.json'));live=Path('/root/ascend-dev/fnx/fnx')
for name,digest in before.items():assert hashlib.sha256((live/name).read_bytes()).hexdigest()==digest,name
assert Path('/root/ascend-dev/p275.sh').read_bytes()==(P/'p275-before.sh').read_bytes()
assert not (live/'norm_fused.py').exists() and not (live/'libfnx_norm_fused.so').exists()
health=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5));assert health['service_revision']=='r144' and health['worker_alive']
(P/'final-health.json').write_text(json.dumps(health,indent=2))
q={};qb=json.load(open(P/'baseline-quality.json'));qc=json.load(open(P/'candidate-quality.json'))
for name,a in qb.items():
    b=qc[name];ds=[i for i,(x,y) in enumerate(zip(a['nll_tokens'],b['nll_tokens'])) if x!=y]
    q[name]=dict(tokens=a['tokens'],targets=a['scored'],baseline_nll=a['mean_nll'],candidate_nll=b['mean_nll'],different_targets=len(ds),different_argmax=sum(x!=y for x,y in zip(a['argmax'],b['argmax'])),first_different_tail_index=ds[0] if ds else None,max_target_nll_abs=max((abs(x-y) for x,y in zip(a['nll_tokens'],b['nll_tokens'])),default=0))
(P/'quality-difference-summary.json').write_text(json.dumps(q,indent=2))
b,c=r['baseline'],r['candidate']
cached_exact=json.load(open(P/'baseline/cached256k.json'))['answer']==json.load(open(P/'candidate/cached256k.json'))['answer']
verification=dict(service_revision='r144',candidate_rejected=True,production_file_hashes_unchanged=len(before),startup_unchanged=True,new_module_not_installed=True,restored_fresh_http_answer_exact=True,private_env_removed=True,lease_removed=True,quality_gate_passed=False,performance_gate_passed=r['speedup_8k_pct']>.5 and r['speedup_256k_pct']>.5,cached_answer_exact=cached_exact)
(P/'verification.json').write_text(json.dumps(verification,indent=2))
table='| 场景 | r144 TPOT, ms | 融合候选 TPOT, ms | 输出速度变化 |\n|---|---:|---:|---:|\n'
for name,kk in [('fresh 8k（三次中位）','median_8k_tpot_ms'),('fresh 256k（各一次）','tpot_256k_ms'),('256k缓存重复（各一次，非流式）','cached_256k_tpot_ms')]:
    table+=f'| {name} | {b[kk]:.2f} | {c[kk]:.2f} | +{(b[kk]/c[kk]-1)*100:.2f}% |\n'
qt='| 冻结文本 | r144 NLL | 候选 NLL | 不同目标 / 199 | 不同argmax |\n|---|---:|---:|---:|---:|\n'
for name,row in q.items():qt+=f"| {name} | {row['baseline_nll']:.6f} | {row['candidate_nll']:.6f} | {row['different_targets']} | {row['different_argmax']} |\n"
repro=json.load(open(P/'rounding-reproducer-summary.json'))
report=f'''# r148 重启A/B：归一化融合更快，但未通过精确性验收

2026-11-12，192.168.18.206，2×Ascend 910B4-1（物理3/5、两进程24+24），CANN9.1.0、驱动26.1.1、torch2.10.0+cpu、torch_npu2.10.0.post4。正式源码起点794a22e / r144，测试前仓库HEAD f53ec03且干净。

结论：候选fresh 8k / 256k输出速度实测提高约**2.42% / 2.55%**，但12k质量样本及一份8k回答出现差异，不符合精确融合门禁，**没有部署候选，正式服务已恢复r144**。不能把此候选记录成已交付提速。

## 整模型HTTP结果

{table}

输出速度变化按`TPOT基线/TPOT候选-1`计算；不是TPOT降低百分比。fresh 8k TPOT降低约{(1-c['median_8k_tpot_ms']/b['median_8k_tpot_ms'])*100:.2f}%，每token省0.80ms；fresh 256k省0.86ms。8k为8064输入+128输出，每版本三份固定输入；256k为262016输入+128输出、每版本一次，不外推256k输入后追加8k输出。缓存重复明确单列，新增prefill=0；不充当fresh prefill收益。

8k HTTP TTFT中位：{b['median_8k_ttft_s']:.3f}→{c['median_8k_ttft_s']:.3f}s。256k HTTP TTFT：{b['ttft_256k_s']:.3f}→{c['ttft_256k_s']:.3f}s，约69.9s，没有可确认的长文TTFT改善。该改动针对decode归一化，不能据微小TTFT波动宣称prefill提速。

测试暂停已识别的FNX watchdog和model-switch管理器，停止原服务后，每次只启动一份完整Engine。r144基线和候选各重新加载，在仅本机的127.0.0.1:19002测量，避开外部业务排队。配置、权重、冻结HTTP输入一致，候选仅启用FNX_NORM_FUSED。两次成功启动到HTTP ready分别{json.load(open(P/'baseline-startup.json'))['start_to_http_ready_s']:.1f} / {json.load(open(P/'candidate-startup.json'))['start_to_http_ready_s']:.1f}s，单列启动时间，不算TTFT。第一轮基线导入acl失败已修复，未计入这些时间或性能数据。

## 精确性门禁失败

{qt}

6k和24k样本逐目标NLL/argmax全部相同；12k有68/199个目标NLL不同，首次可观察差异为尾部索引130（0起算），最大单目标差约{q['service12000']['max_target_nll_abs']:.9f}。三份共597个目标的argmax均相同。12k均值稍降不构成通用质量更好的证据，尤其它是重复服务文本；本轮不能宣称保持精确数值路径。

三份fresh 8k完整回答一致性为{r['answers_8k_exact']}，第一份在第60个字符起有措辞变化，长度266→269；256k fresh回答相同，三事实均3/3，缓存回答一致性为{cached_exact}。没有把措辞变化直接称为事实质量下降，但它进一步证明新路径不再逐位等价。历史100题未重跑，未额外重跑旧200-token loss；质量门禁采用真实plan_prefill/prefill_chunk/finish_prefill与597个尾部decode目标。

本轮完整L00/L12/L24/L44输出、BF16历史及FP32状态对拍曾通过，覆盖decode及128/512/2048行prefill；此前全部36层DeltaNet子路径样本也通过。这些样本仍未覆盖这次全模型分歧，说明局部一致不能替代整模型门禁。

## 已保存的归一化反例

额外算子探针不启动Engine。常规随机/SiLU分布扫描2,097,152行未找到差异；改为稀疏舍入边界输入后，18,432行内找到反例，并回到原单token形状`[1,16,128]`复验。

每行只有四个非零值：索引60/99为2^-12，61为1/16，80为1。关闭query缩放，原CANN平方/ReduceSum/查表路径在索引80输出**0.99609375**，新融合路径输出**1.0**；16行共64个非零输出不同。输入的平方均可精确表示，这证明新的归约融合未保持原数值路径，不能归咎于query缩放。已保存四个小型合成张量及可复现脚本，不含用户会话激活。

这不是从12k失败请求中抓取的真实中间张量，因此尚未逐算子证明它与那次全模型分歧是同一个首次触发点。下一版需保留原CANN归约或复刻其累加/舍入行为，再重走整模型验收，不能因此次性能更快而跳过。

原稀疏搜索记录沿用了旧脚本的`silu`元数据字段，该字段在此不适用；生成器实际为上述稀疏BF16常量。rounding-reproducer-summary.json明确标注，原输入与对拍结果未改。

## 回退和交付范围

候选只存在隔离目录，未安装新Python/ELF或修改正式p275.sh。测试结束由独立恢复流程通过原p275.sh重新启动r144；39个正式Python/ELF与测试前哈希一致，启动脚本逐字节相同。恢复后的fresh 8k HTTP完整回答与r144基线相同，新增prefill8060，端到端{post['elapsed_s']:.3f}s。该验收为非流式接口，stats不含tokens字段，不能将其TPOT直接拿来替代上表的流式基线。

管理器恢复active、RefuseManualStart=no，一份正常watchdog恢复；租约、临时drop-in、恢复timer和完整ENV快照已移除。部署辅助脚本曾准备但没有执行；AB accepted=false禁止了发布路径。最终服务仍为r144，而不是r148。

原始目录：/models/fnx-diagnostics/r148-restart-ab-20261112。证据含全部HTTP统计/回答、质量逐目标数组、输入哈希、完整层对拍、失败原因、合成反例和候选实现归档。只提交文档与实验数据，正式运行代码保持不变。
'''
(P/'report.md').write_text(report)
old=(P/'baseline/fnx/engine.py').read_bytes().decode('utf-8-sig').splitlines(keepends=True)
new=(P/'candidate/fnx/engine.py').read_bytes().decode('utf-8-sig').splitlines(keepends=True)
(P/'engine.patch').write_text(''.join(difflib.unified_diff(old,new,fromfile='a/fnx/engine.py',tofile='b/fnx/engine.py')))
with tarfile.open(B/'rejected-candidate-implementation.tgz','w:gz') as tar:
    for source,name in [(C/'norm_fused.cpp','norm_fused.cpp'),(C/'fnx/norm_fused.py','norm_fused.py'),(C/'fnx/libfnx_norm_fused.so','libfnx_norm_fused.so'),(P/'engine.patch','engine.patch')]:tar.add(source,arcname=name,recursive=False)
names=['ab-summary.json','quality-difference-summary.json','baseline-reference-check.json','input-manifest.json','production-before-hashes.json','verification.json','post-restore-http.json','final-health.json','restored-health.json','setup-corrections.json','answer-difference.json',
       'baseline-quality.json','candidate-quality.json','baseline-health.json','candidate-health.json','baseline-startup.json','candidate-startup.json','layer-probe.json',
       'controller.log','layer-probe.log','baseline-server.log','candidate-server.log','restore.log','rounding-search.log','rounding-sparse.log','rounding-search.json','rounding-sparse-search.json','rounding-reproducer-summary.json',
       'http_bench.py','quality_client.py','evaluate.py','layer_probe.py','launch.py','snapshot.py','stop_owned.py','controller.sh','rescue.sh','find_rounding_case.py','find_rounding_sparse.py','post_restore.py','prefill_nll.py','fixtures.json','engine.patch','package_evidence.py','report.md']
for name in names:shutil.copyfile(P/name,B/name)
for folder in ['baseline','candidate','inputs']:
    for f in (P/folder).iterdir():
        if f.is_file() and f.suffix=='.json':
            target=B/folder/f.name;target.parent.mkdir(exist_ok=True,parents=True);shutil.copyfile(f,target)
for f in P.glob('rounding-sparse-case-*.pt'):shutil.copyfile(f,B/f.name)
(B/'README.md').write_text('''# Reboot A/B of r146 normalization candidate (r148 experiment)

The candidate is faster but rejected: 68 target NLL differences in the 12k fixture and one changed 8k answer. Production was restored to r144. Read report.md, ab-summary.json and verification.json. This commit does not activate or modify production runtime code.

Baseline and candidate each rebooted with the same production configuration, on loopback port 19002 without user traffic. Three frozen fresh 8k inputs, one near-capacity 256k input and a nonstream cached repeat per version. Quality uses three frozen documents and 597 targets through the true prefill policy. Sample counts and scope limits are explicit.

rejected-candidate-implementation.tgz includes the unaccepted C++ kernel, Python wrapper, Linux aarch64 library and engine diff. Do not install it as an exact optimization. Build settings are the same CANN9.1 Bisheng dav-c220-vec -O2 pattern as kernels/l2norm_post/build.sh, with norm_fused.cpp and an explicit output library path. Numeric implementation was unchanged from the r146/r147 component checks.

rounding-sparse-case-*.pt are small synthetic tensors rechecked at original [1,16,128] shape, query scaling disabled. The raw inherited silu field is inapplicable for this sparse generator; use rounding-reproducer-summary.json and the source generator. No user request activations were captured. No weights, graph caches, credentials or full process ENV are included.

Controller and restore scripts are historical operational evidence and depend on fixed server paths plus a fresh private snapshot. The private snapshot was removed after recovery. Do not run the controller on an online service without a maintenance window and renewed site checks. SHA256SUMS covers all distributed files.
''')
files=sorted(f for f in B.rglob('*') if f.is_file() and f.name!='SHA256SUMS')
(B/'SHA256SUMS').write_text(''.join(hashlib.sha256(f.read_bytes()).hexdigest()+'  '+str(f.relative_to(B))+'\n' for f in files))
archive=P/'r148-restart-ab-evidence.tgz'
with tarfile.open(archive,'w:gz') as tar:tar.add(B,arcname='r148-restart-ab-evidence')
print('REJECTED_CANDIDATE_EVIDENCE_READY',len(files),archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest())
