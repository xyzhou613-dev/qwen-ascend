from pathlib import Path
import shutil
r=Path('/root/ascend-dev');s=r/'r93';s.mkdir(exist_ok=True)
shutil.copytree(r/'fnx/fnx',s/'fnx',dirs_exist_ok=True)
p=s/'fnx/pipeline_worker.py';text=p.read_text()
text=text.replace('    peer=None\n','    peer=None\n    profiler=None\n',1)
needle="            if op=='peer_init':";assert text.count(needle)==1
text=text.replace(needle,'''            if op=='prof_start':
                profiler=torch_npu.profiler.profile(
                    activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],
                    record_shapes=True,
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(args[0]))
                profiler.start();value=None
            elif op=='prof_stop':
                profiler.stop();profiler=None;value=None
            elif op=='peer_init':''')
text=text.replace('import torch\n','import torch\nimport torch_npu\n',1)
p.write_text(text);compile(text,str(p),'exec')
env=(r/'r92/candidate.env').read_text()+'FNX_CONV_CACHE=0\n'
(s/'profile.env').write_text(env)
print('R93_PREPARED',flush=True)
