"""定位：v2 解包里到底是哪个算子慢（int16 位运算？uint8 移位？）。"""
import time

import torch
import torch_npu  # noqa: F401

DEV = "npu:0"
N, K = 1280, 2560
k = 10


def bench(fn, n=20, warm=5):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    torch.manual_seed(0)
    p = torch.randint(0, 256, (k, N, K // 2), dtype=torch.uint8, device=DEV)
    q = torch.randint(-8, 8, (k, N, K), dtype=torch.int8, device=DEV)
    s = (torch.rand(k, N, K // 64, dtype=torch.float16, device=DEV) * 0.01)

    print(f"=== 单算子耗时（[10,{N},{K//2}] uint8 = {p.numel()/1e6:.1f}M 元素）===", flush=True)
    cases = [
        ("uint8 -> bf16 cast", lambda: p.to(torch.bfloat16)),
        ("uint8 -> int16 cast", lambda: p.to(torch.int16)),
        ("uint8 & 0x0F", lambda: p & 0x0F),
        ("uint8 >> 4", lambda: p >> 4),
        ("uint8 &0x0F -> int16", lambda: (p & 0x0F).to(torch.int16)),
        ("int16 ^8 -8", lambda: ((p & 0x0F).to(torch.int16) ^ 8) - 8),
        ("int16 -> bf16", lambda: (p & 0x0F).to(torch.int16).to(torch.bfloat16)),
        ("bf16 解包: /16 floor", lambda: torch.floor(p.to(torch.bfloat16) / 16)),
        ("bf16 解包: 全长(4ops)", lambda: (lambda pb: (
            pb - torch.floor(pb / 16) * 16, torch.floor(pb / 16)))(p.to(torch.bfloat16))),
        ("【容器路径】int8->bf16 + mul", lambda: q.to(torch.bfloat16).view(k, N, K // 64, 64)
                                                      * s.to(torch.bfloat16).unsqueeze(-1)),
    ]
    for tag, fn in cases:
        try:
            print(f"  {tag:34s} {bench(fn):8.3f} ms", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"  {tag:34s} 失败 {type(e).__name__}: {str(e)[:60]}", flush=True)

    print("\n=== 纯搬运上限参考 ===", flush=True)
    d2 = torch.empty_like(p)
    print(f"  copy_ 同尺寸 uint8                 {bench(lambda: d2.copy_(p)):8.3f} ms", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
