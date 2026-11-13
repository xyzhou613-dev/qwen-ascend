"""Write final docs only after observed canonical recovery; preserve all old bytes."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

P=Path(__file__).resolve().parent;repo=Path('/root/qwen-ascend')
data=json.load(open(P/'paired-analysis.json'));final=json.load(open(P/'final-validation.json'))
assert final['health']['service_revision']=='r152' and final['private_env_removed']
assert final['restored_within_six_hour_cap']
assert data['nll']['complete'] and data['fresh_natural_prefill']['complete']
assert not data['legacy_natural_regression']['pass_limit'] and not data['passed']
relative='docs/Qwen3.8-Flash-Next-2x910B4-技术文档.md';doc=repo/relative
old=doc.read_bytes();base=subprocess.check_output(['git','show','HEAD:'+relative],cwd=repo)
assert old==base,'preserve someone else\'s uncommitted technical document changes'
assert b'<!-- r157-paired-quality -->' not in old
perf=data['performance'];r=data['legacy_natural_regression'];bad=next(x for x in r['slices'] if x['key']=='natural18000')
paired=data['tasks']['paired'];numeric=json.load(open('/models/fnx-diagnostics/r158-bf16-down-20261113/numeric-result.json'))
block=f'''<!-- r157-paired-quality -->
## r157 整模型对照与质量配对（2026-11-13—14，正式保留 r152）

同轮初始 r152 / W8A8 / W8A8+新down / 回退r152 的 fresh 8k TPOT 分别为 {perf['baseline']['median_8k_tpot_ms']:.2f}/{perf['smooth']['median_8k_tpot_ms']:.2f}/{perf['combined']['median_8k_tpot_ms']:.2f}/{perf['rollback']['median_8k_tpot_ms']:.2f}ms；长文为 {perf['baseline']['median_256k_tpot_ms']:.2f}/{perf['smooth']['median_256k_tpot_ms']:.2f}/{perf['combined']['median_256k_tpot_ms']:.2f}/{perf['rollback']['median_256k_tpot_ms']:.2f}ms，各三次中位数。组合相对初始r152输出速度提高约18.1%/17.8%，新down在W8A8之上额外节省1.42/1.53ms。A/A2回答完全一致；候选没有发布。

100352个目标完整配对平均ΔNLL={data['nll']['delta']:+.6f}nat、28切片最差+0.027416，10个新鲜中英多长度真prefill切片均在门槛内。但旧natural18000切片组合Δ={bad['delta']:+.6f}>+0.05，W8A8单独对照为+0.013832；完整CMMLU为253→252/300，保守95%非劣下界未达门槛。任务最终配对{paired}/1200；状态为{data['status']}，不能将部分任务结果当完整验收。

测试按用户六小时上限恢复并保留断点；正式9002 r152、manager及单份watchdog已验证恢复，45个生产文件/p275未改，临时实验控制和私有ENV清理。新增正常开机恢复服务保留，用固定PCI对修复设备枚举变化，尊重人工停机；多次主机重启原因未确认。本轮没有新的完整PMU账本，旧r145只作历史参考。详见[r157完整记录](r157-combined-quality.md)。

## r158 BF16权重舍入探针（未集成、未作时序及整模型验收）

对新down先补回INT4×scale的BF16权重舍入。单个真实L00 down、512专家、53组输入，最大relative L2由{numeric['old_max_relative_l2']:.9f}降为{numeric['rounded_max_relative_l2']['256']:.9f}，最低逐位相同率约99.9844%；fresh graph/eager通过，峰值张量约492MiB。它没有混入r157配对，尚未证明旧18k回归修复，也不能沿用27.96/28.54ms作为其TPOT。下一步复验旧18k、独立组件/整层耗时及完整质量。详见[r158组件记录](r158-bf16-down.md)。
<!-- /r157-paired-quality -->

'''
payload=block.encode('utf-8')
(P/'techdoc-block.md').write_bytes(payload)
doc.write_bytes(payload+old);assert doc.read_bytes().endswith(old)
for source,target in ((P/'report.md',repo/'docs/r157-combined-quality.md'),
                      (Path('/models/fnx-diagnostics/r158-bf16-down-20261113/README.md'),repo/'docs/r158-bf16-down.md')):
 assert not target.exists(),target
 shutil.copyfile(source,target)
proof=dict(path=relative,before_sha256=hashlib.sha256(old).hexdigest(),after_sha256=hashlib.sha256(doc.read_bytes()).hexdigest(),old_bytes_preserved_as_suffix=True,old_bytes=len(old),prefix_bytes=len(payload))
(P/'technical-doc-update.json').write_text(json.dumps(proof,indent=2)+'\n')
print('FINAL_DOCUMENTS_WRITTEN_WITH_OLD_BYTES_PRESERVED')
