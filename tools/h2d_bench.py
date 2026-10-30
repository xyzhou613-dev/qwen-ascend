"""★ H2D 代价实测：判断 mmap / 分层驻留在我们这台机上是否成立。

要回答的核心问题：**每 token 从宿主流入"K 层的 top-10 专家"要多久？**

一个 4bit 打包专家的大小（本模型实测形状）：
  gate_up_proj [1280, 2560] /2 = 1.638 MB
  down_proj   [2560,  640] /2 = 0.819 MB
  → 每专家 2.458 MB，每层 top-10 = 24.58 MB

测六种模式：
  A 纯 H2D，连续 24.58 MB，pageable
  B 纯 H2D，连续 24.58 MB，pinned
  C 10 × 2.458 MB 分段 H2D（pinned）—— "每专家一个 blob" 的设计
  D 宿主 index_select（随机 10 个专家）后一次 H2D（pinned）—— 旧方案的模式
  E C 的异步双缓冲流水版（先发再等）
  F 整层 1.48 GiB（pinned）—— 看大块带宽上限
"""
import time

import torch
import torch_npu  # noqa: F401

DEV = "npu:0"
MB = 1 << 20
E_GU = 1280 * 2560 // 2          # 1.638 MB（4bit 打包）
E_DN = 2560 * 640 // 2           # 0.819 MB
PER_EXP = E_GU + E_DN            # 2.458 MB
LAYER10 = PER_EXP * 10           # 24.58 MB


def bench(fn, n=20, warm=3):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t0) / n


def rep(tag, dt, nbytes):
    gbs = nbytes / dt / 1e9 if dt > 0 else 0
    print(f"  {tag:44s} {dt*1e3:8.2f} ms   {gbs:7.2f} GB/s", flush=True)
    return gbs


def main():
    torch.manual_seed(0)
    print("=== H2D 实测（每专家 %.3f MB，每层 top-10 = %.2f MB）==="
          % (PER_EXP / MB, LAYER10 / MB), flush=True)

    # ---------- A/B：连续块 ----------
    for tag, pin in (("A 连续 24.58MB pageable", False), ("B 连续 24.58MB pinned", True)):
        h = torch.empty(LAYER10, dtype=torch.uint8)
        if pin:
            h = h.pin_memory()
        d = torch.empty(LAYER10, dtype=torch.uint8, device=DEV)

        def f(h=h, d=d):
            d.copy_(h, non_blocking=True)

        rep(tag, bench(f), LAYER10)

    # ---------- C：10 段连续（每专家一个 blob，pinned）----------
    hs = [torch.empty(PER_EXP, dtype=torch.uint8).pin_memory() for _ in range(512)]
    ds = [torch.empty(PER_EXP, dtype=torch.uint8, device=DEV) for _ in range(10)]
    ids = list(range(10))

    def fc():
        for j, i in enumerate(ids):
            ds[j].copy_(hs[i], non_blocking=True)

    gb_c = rep("C 10×2.458MB 分段 pinned", bench(fc), LAYER10)
    del ds

    # ---------- D：宿主 index_select 后一次 H2D ----------
    pool = torch.empty(512 * PER_EXP // 8, dtype=torch.uint8)      # 只取 1/8 控制内存
    pool = pool.pin_memory() if hasattr(torch, "empty") else pool
    per = pool.numel() // 512
    idx = torch.tensor([7, 3, 511, 100, 42, 8, 200, 33, 77, 5], dtype=torch.long)
    dsel = torch.empty(per * 10, dtype=torch.uint8, device=DEV)

    def fd():
        sub = pool.view(512, per)[idx].reshape(-1)
        dsel.copy_(sub, non_blocking=True)

    rep("D 宿主 index_select(10/512)+H2D pinned", bench(fd), per * 10)
    del pool, dsel

    # ---------- E：流水（先全发，再同步一次）----------
    d2 = [torch.empty(PER_EXP, dtype=torch.uint8, device=DEV) for _ in range(10)]

    def fe():
        for j, i in enumerate(ids):
            d2[j].copy_(hs[i], non_blocking=True)

    rep("E 同上但一次性发完再同步（流水）", bench(fe, n=50), LAYER10)
    del d2

    # ---------- F：整层 1.48 GiB pinned ----------
    LAYER_FULL = 1_587_000_000
    try:
        hfull = torch.empty(LAYER_FULL, dtype=torch.uint8).pin_memory()
        dfull = torch.empty(LAYER_FULL, dtype=torch.uint8, device=DEV)

        def ff():
            dfull.copy_(hfull, non_blocking=True)

        rep("F 整层 1.48GiB pinned", bench(ff, n=3, warm=1), LAYER_FULL)
        del hfull, dfull
    except Exception as e:  # noqa: BLE001
        print(f"  F 跳过（内存不够）: {type(e).__name__}: {str(e)[:80]}")

    # ---------- 换算：每 token 流式 K 层的代价 ----------
    print("\n=== 换算：每 token 从宿主流式 K 层 top-10 专家 ===", flush=True)
    for gb, name in ((gb_c, "C 分段"),):
        print(f"  按 {name} 的 {gb:.2f} GB/s：")
        for K in (2, 4, 6, 10, 16, 23):
            ms = K * LAYER10 / (gb * 1e9) * 1e3
            print(f"    流式 {K:2d} 层 = {K*LAYER10/MB:7.1f} MB -> {ms:6.2f} ms/token")
    print("\n  对照：当前线上 318 ms/token；若流式 6 层只花 ~7 ms，则完全值得")


if __name__ == "__main__":
    main()
