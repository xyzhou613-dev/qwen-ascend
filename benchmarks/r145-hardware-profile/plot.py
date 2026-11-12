import json,collections
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
P=Path(__file__).resolve().parent
colors={'dense_index_matmul':'#2878a5','quantized_matmul':'#dc8931','expert_gather':'#b5699b','fp32_deltanet':'#579c69','other':'#7879ae'}
def group(n):
 n=n.lower()
 if 'weightquant' in n:return 'quantized_matmul'
 if 'matmul' in n:return 'dense_index_matmul'
 if 'gather_pair' in n:return 'expert_gather'
 if 'delta_' in n:return 'fp32_deltanet'
 return 'other'
fig,axs=plt.subplots(2,1,figsize=(14,7),layout='constrained')
records=json.load(open(P/'runs.json'))
for ax,context in zip(axs,['8k','256k']):
 rec=next(r for r in records if r['tag']==context+'_timing');lo=rec['rows'][7]['end_ns'];hi=rec['rows'][9]['end_ns']
 d=json.load(open(P/'aligned'/f'{context}_timing.json'));origin=d['epoch_origin_ns'];window=[e for e in d['traceEvents'] if e.get('ph')=='M'];bars=collections.defaultdict(list)
 window.append(dict(ph='M',name='process_name',pid=20,args={'name':'Token completion boundaries'}))
 for e in d['traceEvents']:
  if e.get('ph')!='X':continue
  a=origin+e['ts']*1000;b=a+e['dur']*1000
  if b<=lo or a>=hi:continue
  copy=dict(e);copy['ts']=(a-lo)/1000;window.append(copy)
  start=(max(a,lo)-lo)/1e6;width=(min(b,hi)-max(a,lo))/1e6
  if e['pid'] in (10,11):
   y=1 if e['pid']==10 else 0;bars[(y,colors[group(e['name'])])].append((start,width))
  elif e['pid'] in (0,1):
   y=3 if e['pid']==0 else 2;c='#d8dce2' if 'wait' in e['name'] else '#a6b7c7';bars[(y,c)].append((start,width))
 for (y,color),segments in bars.items():ax.broken_barh(segments,(y-.28,.56),facecolors=color,linewidth=0)
 for r in rec['rows'][8:10]:
  x=(r['end_ns']-lo)/1e6;ax.axvline(x,color='#394653',lw=.8,ls='--');ax.text(x-.3,3.5,f"token {r['seq']}",ha='right',fontsize=8)
 ax.set(xlim=(0,(hi-lo)/1e6),ylim=(-.6,3.85),yticks=[0,1,2,3],yticklabels=['NPU 1 kernels','NPU 0 kernels','Host 1 (includes waits)','Host 0 (includes waits)'],xlabel='Time from window start (ms)',title=f'{context}: two consecutive decode tokens; synchronized epoch clocks')
 ax.grid(axis='x',alpha=.15)
 (P/'aligned'/f'{context}_two_tokens.json').write_text(json.dumps(dict(traceEvents=window,displayTimeUnit='ms',epoch_origin_ns=lo)))
fig.legend(handles=[Patch(color=c,label=k.replace('_',' ')) for k,c in colors.items()],loc='outside lower center',ncol=5,frameon=False)
fig.savefig(P/'aligned'/'r144-dual-process-timeline.png',dpi=180)
print('TIMELINE_PNG_AND_COMPACT_TRACE_READY')
