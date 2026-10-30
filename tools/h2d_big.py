"""验证大块 pinned 分配 + 长时间稳定 H2D 带宽（mmap/分层驻留的可行性前提）。"""
import time

import torch
import torch_npu  # noqa: F401

DEV = "npu:0"
GB = 1 << 30


def main():
    print("=== 大块 pinned 分配 + 稳定带宽 ===", flush=True)
    for gb in (2, 8, 24):
        n = gb * GB
        try:
            t0 = time.perf_counter()
            h = torch.empty(n, dtype=torch.uint8).pin_memory()
            t_pin = time.perf_counter() - t0
            d = torch.empty(n, dtype=torch.uint8, device=DEV)
            # 预热一次
            d.copy_(h, non_blocking=True)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            reps = 3
            for _ in range(reps):
                d.copy_(h, non_blocking=True)
            torch.npu.synchronize()
            dt = (time.perf_counter() - t0) / reps
            print(f"  {gb:2d} GiB pinned: 锁页耗时 {t_pin:6.2f}s   H2D {dt*1e3:8.2f} ms  "
                  f"{n/dt/1e9:6.2f} GB/s", flush=True)
            del h, d
        except Exception as e:  # noqa: BLE001
            print(f"  {gb:2d} GiB pinned: 失败 {type(e).__name__}: {str(e)[:90]}", flush=True)

    print("\n=== mmap 页缓存 → 设备（模拟 mmap 方案的首次/热读）===", flush=True)
    import mmap
    import os
    p = "/root/ascend-dev/h2d_mmap_probe.bin"
    n = 1 * GB
    with open(p, "wb") as f:
        f.chmod(0o600) if hasattr(f, "chmod") else None
        f.write(b"\0" * n)
    m = mmap.mmap(os.open(p, os.O_RDONLY), n, access=mmap.ACCESS_READ)
    d = torch.empty(n, dtype=torch.uint8, device=DEV)
    # 从 mmap 读到一个普通张量再上卡（等价于"文件页缓存 → 设备"）
    buf = torch.frombuffer(bytearray(m), dtype=torch.uint8)
    for tag in ("冷(首次触碰页缓存)", "热(页缓存命中)"):
        t0 = time.perf_counter()
        buf.copy_(torch.frombuffer(bytearray(m), dtype=torch.uint8))
        torch.npu.synchronize()
        t_cpu = time.perf_counter() - t0
        t0 = time.perf_counter()
        d.copy_(buf, non_blocking=True)
        torch.npu.synchronize()
        dt = time.perf_counter() - t0
        print(f"  {tag:18s} 宿主逐字节拷贝 {t_cpu*1e3:7.2f} ms  H2D {dt*1e3:7.2f} ms "
              f"({n/dt/1e9:.2f} GB/s)", flush=True)
    del m
    os.remove(p)


if __name__ == "__main__":
    main()
