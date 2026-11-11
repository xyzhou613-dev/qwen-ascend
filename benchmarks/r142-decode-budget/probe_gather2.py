"""Locate the gather mismatch: torch reference vs each kernel, and repeatability."""
import ctypes
import torch
import torch_npu

P = '/models/fnx-diagnostics/r142-budget-20261111/'
SIG = [ctypes.c_void_p] * 5 + [ctypes.c_int] * 3 + [ctypes.c_void_p]


def load(name):
    lib = ctypes.CDLL(P + name)
    lib.fnx_gather_pair.argtypes = SIG
    lib.fnx_gather_pair.restype = ctypes.c_int
    return lib


def ref(w, s, ids, K, G, N):
    ow = torch.cat([w[e] for e in ids.tolist()], dim=1)          # [K, 10*N/8]
    os_ = torch.cat([s[e] for e in ids.tolist()], dim=1)         # [G, 10*N]
    return ow.contiguous(), os_.contiguous()


for shape in ((2560, 40, 1280), (640, 10, 2560)):
    K, G, N = shape
    E = 512
    dev = 'npu:0'
    g = torch.Generator(device='cpu').manual_seed(7)
    w = torch.randint(-2 ** 31, 2 ** 31 - 1, (E, K, N // 8), dtype=torch.int32, generator=g).npu()
    s = (torch.rand(E, G, N, generator=g, dtype=torch.float32) * 0.02).to(torch.bfloat16).npu()
    ids = torch.tensor([3, 11, 5, 40, 2, 63, 17, 9, 22, 31], dtype=torch.int64).npu()
    rw, rs = ref(w, s, ids, K, G, N)
    stream = torch.npu.current_stream().npu_stream
    for tag, name in (('deployed', 'libfnx_expert_gather_deployed.so'), ('rebuilt', 'libfnx_expert_gather_orig.so'), ('db', 'libfnx_expert_gather_db.so')):
        lib = load(name)
        ow = torch.zeros((K, 10 * N // 8), dtype=torch.int32, device=dev)
        osc = torch.zeros((G, 10 * N), dtype=torch.bfloat16, device=dev)
        outs = []
        for _ in range(3):
            ow.zero_(); osc.zero_()
            lib.fnx_gather_pair(w.data_ptr(), s.data_ptr(), ids.data_ptr(), ow.data_ptr(),
                                osc.data_ptr(), K, G, N, stream)
            torch.npu.synchronize()
            outs.append((ow.clone(), osc.clone()))
        w_ok = [torch.equal(o[0], rw) for o in outs]
        s_ok = [torch.equal(o[1], rs) for o in outs]
        det = all(torch.equal(outs[0][0], o[0]) and torch.equal(outs[0][1], o[1]) for o in outs)
        print(f"K={K} {tag}: weights_ref={w_ok} scales_ref={s_ok} deterministic={det}", flush=True)
print('PROBE2_DONE', flush=True)
