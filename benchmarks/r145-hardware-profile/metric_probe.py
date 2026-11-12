"""Validate installed PMU fields on a small NPUGraph; never load Engine."""
import json
from pathlib import Path
import torch,torch_npu
from torch_npu.profiler import _ExperimentalConfig,ProfilerLevel,AiCMetrics,ExportType
P=Path(__file__).resolve().parent
torch.npu.set_device(0)
x=torch.randn(1,256,device='npu:0',dtype=torch.bfloat16)
w=torch.randn(2048,256,device='npu:0',dtype=torch.bfloat16)
def fn():return torch.nn.functional.linear(x,w).float().square().sum(-1)
fn();torch.npu.synchronize();g=torch.npu.NPUGraph()
with torch.npu.graph(g):out=fn()
for name in ['AiCoreNone','PipeUtilization','Memory','MemoryAccess','L2Cache']:
 config=_ExperimentalConfig(profiler_level=ProfilerLevel.Level1,aic_metrics=getattr(AiCMetrics,name),l2_cache=name=='L2Cache',export_type=[ExportType.Text,ExportType.Db])
 with torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(P/'metric-probe'/name)),experimental_config=config):
  for _ in range(3):g.replay()
  torch.npu.synchronize()
 print('METRIC_EXPORTED',name,flush=True)
print('METRIC_PROBE_COMPLETE',flush=True)
