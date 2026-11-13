"""Diagnostic-only clocked host spans and separate PMU metric groups."""
import contextlib,json,os,time
from pathlib import Path
import torch,torch_npu
_enabled=False
_torch=False
_tag=''
_seq=-1
_events=[]
def set_seq(value):
 global _seq
 _seq=value
def begin(tag,profiled):
 global _enabled,_torch,_tag,_events
 _enabled=True;_torch=profiled;_tag=tag;_events=[]
def finish(path):
 global _enabled,_torch
 _enabled=False;_torch=False
 Path(path).write_text(json.dumps(_events));return len(_events)
@contextlib.contextmanager
def span(name):
 if not _enabled:
  yield;return
 a=time.time_ns();ma=time.perf_counter_ns()
 if _torch:
  with torch.profiler.record_function(f'r145/{_tag}/{_seq}/{name}'):yield
 else:yield
 _events.append(dict(name=name,seq=_seq,pid=os.getpid(),start_ns=a,end_ns=time.time_ns(),mono_start_ns=ma,mono_end_ns=time.perf_counter_ns()))
def make_profiler(path,metric):
 from torch_npu.profiler import _ExperimentalConfig,ProfilerLevel,AiCMetrics,ExportType
 return torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,torch_npu.profiler.ProfilerActivity.NPU],record_shapes=True,
  on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
  experimental_config=_ExperimentalConfig(profiler_level=ProfilerLevel.Level1,aic_metrics=getattr(AiCMetrics,metric),l2_cache=metric=='L2Cache',export_type=[ExportType.Text,ExportType.Db]))
