import time, torch, torch_npu
E, N, K = 512, 1280, 2560
q = torch.randint(-8, 8, (E, N, K), dtype=torch.int8, device='npu:0')
torch.npu.synchronize()
t = time.time()
q16 = q.to(torch.int16)
q32 = q16.view(torch.int32)
p = (q32 & 0x0F) | (((q32 >> 16) & 0x0F) << 4)
p8 = (p - ((p >= 128).to(torch.int32) << 8)).to(torch.int8)
torch.npu.synchronize()
print(f'npu int32-view pack {time.time()-t:6.3f}s  shape {tuple(p8.shape)}', flush=True)

idx = torch.arange(10, device='npu:0')
t = time.time()
sp = p8.index_select(0, idx)
p16 = sp.to(torch.int16) & 0xFF
lo = ((p16 & 0x0F) ^ 8) - 8
hi = (((p16 >> 4) & 0x0F) ^ 8) - 8
rec = torch.stack([lo, hi], -1).reshape(10, N, K)
print('unpack == 原 int8 :', torch.equal(rec, q[:10]), flush=True)
torch.npu.synchronize()
print(f'unpack 10 experts  {time.time()-t:6.3f}s', flush=True)

s = (torch.rand(10, N, K // 32, dtype=torch.float32, device='npu:0') * 0.01).to(torch.float16)
t = time.time()
qb = rec.to(torch.bfloat16).view(10, N, K // 32, 32)
w = (qb * s.to(torch.bfloat16).unsqueeze(-1)).reshape(10, N, K)
torch.npu.synchronize()
print(f'deq 10 experts     {time.time()-t:6.3f}s', flush=True)
print('OK')
