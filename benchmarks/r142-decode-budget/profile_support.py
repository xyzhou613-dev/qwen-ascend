from pathlib import Path
import torch, torch_npu


def tensor(t):
    return dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                bytes=t.numel() * t.element_size(), storage_bytes=t.untyped_storage().nbytes())


def describe(o):
    if o is None:
        return None
    if isinstance(o, torch.Tensor):
        return dict(kind='Tensor', **tensor(o))
    d = dict(kind=type(o).__name__)
    for n in ('K', 'N', 'E', 'G', 'gs', 'expert_major'):
        if hasattr(o, n):
            d[n] = getattr(o, n)
    for n in ('w', 'q', 's', 'w8t', 'sc', 'w4', '_scales'):
        t = getattr(o, n, None)
        if isinstance(t, torch.Tensor):
            d[n] = tensor(t)
    chunks = getattr(o, 'chunks', None)
    if chunks:
        d['chunks'] = [dict(weight=tensor(w), scale=tensor(s), begin=a, end=b) for w, s, a, b in chunks]
    return d


def inventory(m):
    attrs = ('a_di', 'a_down', 'a_up', 'a_inj', 'm_di', 'm_down', 'm_up', 'm_inj', 'w_qkvz', 'w_qkv', 'w_z',
             'w_ab', 'w_b', 'w_a', 'w_out', 'w_qkvf', 'w_q', 'w_k', 'w_v', 'w_o', 'index_w', 'sh_gu', 'sh_down',
             'sh_gate', 'sh_up', 'sh_gate2', 'gate_w', 'exp_gu', 'exp_dn')
    return dict(layers=[dict(index=L.idx, type=L.ltype,
                             projections={n: describe(getattr(L, n, None)) for n in attrs if getattr(L, n, None) is not None})
                        for L in m.layers0],
                head={n: describe(getattr(m, n, None)) for n in ('lm_head', 'mix_down', 'mix_up')}, ctx=m.P)


def make_profiler(path, metric):
    from torch_npu.profiler import _ExperimentalConfig, ProfilerLevel, AiCMetrics, ExportType
    metrics = {'timing': AiCMetrics.AiCoreNone, 'memory': AiCMetrics.MemoryAccess,
               'arithmetic': AiCMetrics.ArithmeticUtilization}
    return torch_npu.profiler.profile(activities=[torch_npu.profiler.ProfilerActivity.CPU,
                                                  torch_npu.profiler.ProfilerActivity.NPU],
                                      record_shapes=True,
                                      on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(path),
                                      experimental_config=_ExperimentalConfig(profiler_level=ProfilerLevel.Level1,
                                                                             aic_metrics=metrics[metric],
                                                                             export_type=[ExportType.Text,
                                                                                          ExportType.Db]))
