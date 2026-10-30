"""★★ op 级微基准（第 34 轮）：动生产代码之前，先在真实形状上筛候选实现。

背景：msProf 显示反量化逐元素链占 51%，但直接替换实测**更慢**
（`div(rounding_mode='floor')` 和广播 `addcmul` 都把单个请求拖死）。
所以定规矩：**任何进热路径的算子替换，必须先在这里过闸**。

用法（容器内）：
  python tools/op_bench.py            # 每个候选 5 次
  python tools/op_bench.py 20         # 每个候选 20 次
只分配几十 MB 显存，**可以和线上服务共存**。
"""
import sys
import time

import torch
import torch_npu  # noqa: F401

N = 1280          # 每层每投影的输出行数（gate_up）
K = 2560          # 输入维度
KK = K // 2
KE = 10           # top-10 专家
S = K // 128      # g128 下每 128 个 k 一个 scale
DEV = "npu:0"
REP = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def tm(fn, n=REP, warm=1):
    for _ in range(warm):
        fn()
    torch.npu.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.npu.synchronize()
    return (time.perf_counter() - t) / n * 1e3


def bench(title, cases, note=""):
    print(f"\n=== {title} ==={note}", flush=True)
    base_name, base_t = None, None
    for name, fn in cases:
        try:
            t = tm(fn)
        except Exception as ex:                      # noqa: BLE001
            print(f"  {name:34s} 失败: {type(ex).__name__}: {str(ex)[:60]}", flush=True)
            continue
        if base_t is None:
            base_name, base_t = name, t
            print(f"  {name:34s} {t:8.3f} ms   （基准）", flush=True)
        else:
            d = (t - base_t) / base_t * 100
            flag = "✅ 快" if t < base_t * 0.97 else ("❌ 慢" if t > base_t * 1.03 else "≈ 持平")
            print(f"  {name:34s} {t:8.3f} ms   {d:+6.1f}%  vs {base_name}  {flag}", flush=True)


def main():
    torch.manual_seed(0)
    pb = torch.randint(0, 255, (KE, N, KK), dtype=torch.uint8, device=DEV).to(torch.bfloat16)
    lo = torch.randint(0, 16, (KE, N, KK), dtype=torch.uint8, device=DEV).to(torch.bfloat16)
    s1 = (torch.rand(KE, N, S, 1, device=DEV) * 0.05 + 0.01).to(torch.bfloat16)
    s64 = s1.expand(KE, N, S, 64).contiguous()
    lo4 = lo.view(KE, N, S, 64)            # ③④ 里必须用 4D 视图，否则广播对不上
    print(f"形状：pb/lo=[{KE},{N},{KK}] s=[{KE},{N},{S},1]  每个候选 {REP} 次", flush=True)

    # ---------- 1) 取高 4 位（byte/16 向下取整） ----------
    bench("① hi = floor(byte/16)", [
        ("floor(pb*0.0625)  【生产】", lambda: torch.floor(pb * 0.0625)),
        ("div(pb,16,rounding_mode=floor)", lambda: torch.div(pb, 16.0, rounding_mode="floor")),
        ("floor_divide(pb,16)", lambda: torch.floor_divide(pb, 16.0)),
        ("pb.div(16.0).floor_()", lambda: pb.div(16.0).floor_()),
    ])

    # ---------- 2) 取低 4 位 ----------
    hi = torch.floor(pb * 0.0625)
    bench("② lo = byte - hi*16", [
        ("pb - hi*16  【生产】", lambda: pb - hi * 16.0),
        ("fmod(pb,16)", lambda: torch.fmod(pb, 16.0)),
        ("remainder(pb,16)", lambda: torch.remainder(pb, 16.0)),
    ])

    # ---------- 3) offset-binary 的 -8 与 scale 合并 ----------
    bench("③ (n'-8)*s：怎么合最省", [
        ("(lo-8)*s  【生产：2 op】", lambda: (lo4 - 8.0) * s1),
        ("addcmul(-8s, lo, s)", lambda: torch.addcmul(-8.0 * s1, lo4, s1)),
        ("lo*s - 8s", lambda: lo4 * s1 - 8.0 * s1),
        ("lo*s - 8*s1（预置 8s）", lambda: lo4 * s1 - s1 * 8.0),
    ])

    # ---------- 4) 单独看这些"替代"算子的绝对代价（解释为什么慢） ----------
    bench("④ 纯算子开销（同一形状，单 op）", [
        ("mul（对照）", lambda: lo4 * s1),
        ("sub（对照）", lambda: lo - 8.0),
        ("addcmul（广播）", lambda: torch.addcmul(-8.0 * s1, lo4, s1)),
        ("div rounding_mode=floor", lambda: torch.div(pb, 16.0, rounding_mode="floor")),
        ("floor", lambda: torch.floor(pb)),
    ], note="  ← 用来判断「少一个 kernel」到底值不值")

    print("\n判读：只有当某候选**稳定快 ≥3%** 时才值得进生产；"
          "单 op 对比里若'替代算子'本身就慢，那它再怎么合并都不会更快。", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
