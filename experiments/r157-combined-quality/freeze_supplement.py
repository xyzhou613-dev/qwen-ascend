"""Prespecified extra structure cases and fresh bilingual true-prefill inputs."""
import hashlib,json,os,random,re,time
from pathlib import Path
P=Path(__file__).resolve().parent
assert not (P/'combined-start.txt').exists(),'freeze supplements before candidate execution'
extra=[];rng=random.Random(15720261113)
for i in range(100):
 data=[dict(id=f'X{i:03d}-{j}',value=rng.randrange(1,99),active=rng.choice([True,False]),group=rng.choice(['A','B','C'])) for j in range(3+i%9)]
 if i%2==0:
  threshold=20+i%40;expected=dict(request_id=f'EX{i:03d}',selected=sorted(r['id'] for r in data if r['active'] and r['value']>=threshold),total=sum(r['value'] for r in data if r['active']))
  instruction=f'Return only a JSON object with request_id, selected (sorted ids where active is true and value >= {threshold}), and total (sum of value for all active records).'
 else:
  expected=dict(request_id=f'EX{i:03d}',groups=[dict(name=g,count=sum(r['group']==g and r['active'] for r in data),total=sum(r['value'] for r in data if r['group']==g and r['active'])) for g in ['A','B','C']])
  instruction='只输出JSON对象，字段request_id照抄；groups按A、B、C排序，每项字段name、count、total，统计该组active=true的记录数和value之和，空组仍保留。'
 extra.append(dict(id=f'structure-extra/{i}',slice='json_extract_extra',kind='json',prompt=instruction+'\n'+json.dumps(dict(request_id=f'EX{i:03d}',records=data),ensure_ascii=False),expected=expected,max_tokens=256))
target=P/'quality-extra-tasks.jsonl';assert not target.exists()
target.write_bytes(''.join(json.dumps(r,ensure_ascii=False,separators=(',',':'))+'\n' for r in extra).encode())
policy=dict(created_epoch=time.time(),base_tasks=1100,extra_structure_tasks=100,total_tasks=1200,
 extra_input_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),reason='JSON domain has 200 cases; avoid declaring a 2pp noninferiority margin from only 100 zero-discordance cases',
 mean_nll_limit=.02,slice_nll_limit=.05,nll_slices='28 prespecified slices: each domain split into 7 consecutive groups of 7 frozen cases',
 task_margin=.02,task_confidence=.95,task_lower_bound='conservative paired improvement lower CP minus worsening upper CP; split alpha when both bounds estimated',
 task_domains=['cmmlu','mmlu_pro','math','code','json'],subbenchmark_point_regression_review_threshold=.02,
 natural_prefill_slice_nll_limit=.05,quality_gate_requires_complete_pairs=True)
(P/'acceptance-policy.json').write_text(json.dumps(policy,indent=2));print('EXTRA_STRUCTURE_FROZEN',len(extra),policy['extra_input_sha256'])
