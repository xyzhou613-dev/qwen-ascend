"""★ 验证 torch_npu.npu_rms_norm 能否加速我们的分组 rms_norm。

关键事实（本脚本第一版踩到的）：`engine.rms_norm(x, w, gs=H)` 里
  - **归一化**按每 H=2560 个元素一组（grouped RMSNorm）；
  - **权重 w 是 10240 维**（每个 hyper-connection 槽位一个），在 flatten 之后按元素乘。
所以 gamma 不能折进 w。可行写法：
  out = npu_rms_norm(x.view(...,-1,H), gamma=ones(H))  # 1 个融合 kernel
  out = out.flatten(-2) * (1+w)                        # 1 个 mul（w 预先缓存）
"""
import time

import torch
import torch_npu  # noqa: F401

from fnx import engine as E

H, HC, EPS = E.H, E.HC, E.EPS
_GC = {}


def ours(x, w):
    return E.rms_norm(x, w, gs=H)


def fused(x, w, gam=True):
    key = w.data_ptr()
    if key not in _GC:
        _GC[key] = (1.0 + w.float())                      # 预缓存：只算一次
    gamma = _GC[key]
    xg = x.view(*x.shape[:-1], -1, H)
    if gam:
        out, _ = torch_npu.npu_rms_norm(xg, torch.ones(H, device=x.device,
                                                       dtype=torch.float32), EPS)
    else:
        out, _ = torch_npu.npu_rms_norm(xg, None, EPS)
    return (out.flatten(-2) * gamma).type_as(x)


def rel(a, b):
    d = (a.float() - b.float()).abs()
    return float(d.mean() / b.float().abs().mean().clamp_min(1e-12)), float(d.max())


def main():
    dev = "npu:0"
    torch.manual_seed(0)
    w = torch.randn(E.HCH, device=dev, dtype=torch.bfloat16) * 0.1
    x = torch.randn(1, 1, E.HCH, device=dev, dtype=torch.bfloat16)

    print("=== 等价性 ===")
    for gam in (True, False):
        try:
            r, mx = rel(fused(x, w, gam), ours(x, w))
            print(f"  gamma={'ones' if gam else 'None':5s} fused vs ours: rel {r*100:8.4f}%  "
                  f"max {mx:.4e}  -> {'等价 ✅' if r < 5e-3 else '有偏差 ⚠️'}")
        except Exception as e:
            print(f"  gamma={'ones' if gam else 'None':5s} 失败: {type(e).__name__}: {e}")

    xf = torch.randn(1, 1, E.HCH, device=dev, dtype=torch.float32)
    r, mx = rel(fused(xf, w), ours(xf, w))
    print(f"  fp32 输入 fused vs ours: rel {r*100:8.4f}%  max {mx:.4e}")

    print("\n=== 计时（200 次）===")
    for name, fn in (("ours", ours), ("fused(ones)", lambda a, b: fused(a, b, True))):
        try:
            for _ in range(10):
                fn(x, w)
            torch.npu.synchronize()
            t0 = time.perf_counter()
            for _ in range(200):
                fn(x, w)
            torch.npu.synchronize()
            print(f"  {name:12s} {((time.perf_counter()-t0)/200)*1e6:8.1f} us/次")
        except Exception as e:
            print(f"  {name:12s} 失败: {e}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
