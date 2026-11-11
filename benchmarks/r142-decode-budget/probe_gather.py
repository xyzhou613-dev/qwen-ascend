"""A/B the double-buffered gather_pair against the deployed one."""
import ctypes
import time
import torch
import torch_npu

P = '/models/fnx-diagnostics/r142-budget-20261111/'
SIG = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]


def load(name):
    lib = ctypes.CDLL(P + name)
    lib.fnx_gather_pair.argtypes = SIG
    lib.fnx_gather_pair.restype = ctypes.c_int
    return lib


def run_case(shape, ids):
    K, G, N = shape
    E = 64
    dev = 'npu:0'
    g = torch.Generator(device='cpu').manual_seed(7)
    w = torch.randint(-2 ** 31, 2 ** 31 - 1, (E, K, N // 8), dtype=torch.int32, generator=g).npu()
    s = (torch.rand(E, G, N, generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16).npu()
    ow = torch.zeros((K, 10 * N // 8), dtype=torch.int32, device=dev)
    os_ = torch.zeros((G, 10 * N), dtype=torch.bfloat16, device=dev)
    stream = torch.npu.current_stream().npu_stream
    out = {}
    for tag, lib in (('orig', LIBA), ('db', LIBB)):
        ow.zero_(); os_.zero_()
        rc = lib.fnx_gather_pair(w.data_ptr(), s.data_ptr(), ids.data_ptr(), ow.data_ptr(),
                                 os_.data_ptr(), K, G, N, stream)
        assert rc == 0, rc
        torch.npu.synchronize()
        out[tag] = (ow.clone(), os_.clone())
    same_w = torch.equal(out['orig'][0], out['db'][0])
    same_s = torch.equal(out['orig'][1], out['db'][1])
    res = {}
    for tag, lib in (('orig', LIBA), ('db', LIBB)):
        for _ in range(20):
            lib.fnx_gather_pair(w.data_ptr(), s.data_ptr(), ids.data_ptr(), ow.data_ptr(),
                                os_.data_ptr(), K, G, N, stream)
        torch.npu.synchronize()
        t0 = time.perf_counter()
        reps = 300
        for _ in range(reps):
            lib.fnx_gather_pair(w.data_ptr(), s.data_ptr(), ids.data_ptr(), ow.data_ptr(),
                                os_.data_ptr(), K, G, N, stream)
        torch.npu.synchronize()
        res[tag] = (time.perf_counter() - t0) / reps * 1e6
    print(f"K={K} G={G} N={N} exact_w={same_w} exact_s={same_s} "
          f"orig={res['orig']:.2f}us db={res['db']:.2f}us "
          f"speedup={res['orig']/res['db']:.3f}x", flush=True)
    return res, same_w and same_s


LIBA = load('libfnx_expert_gather_orig.so')
LIBB = load('libfnx_expert_gather_db.so')
ok = True
tot = {'orig': 0.0, 'db': 0.0}
for shape in ((2560, 40, 1280), (640, 10, 2560)):
    ids = torch.randperm(64)[:10].to(torch.int64).npu().contiguous()
    r, same = run_case(shape, ids)
    ok = ok and same
    for k in tot:
        tot[k] += r[k]
print(f"TOTAL_PER_LAYER orig={tot['orig']:.2f}us db={tot['db']:.2f}us "
      f"save={tot['orig']-tot['db']:.2f}us  x48={48*(tot['orig']-tot['db'])/1000:.3f}ms", flush=True)
print('GATHER_AB_EXACT' if ok else 'GATHER_AB_MISMATCH', flush=True)
