"""14.3-B 的布局方案验证（纯 numpy，不动 kernel）。

问题：现在 kernel 每 segment 要把 x 拆成偶/奇两份（xE=x[0::2], xO=x[1::2]），
      设备侧拆分实测 7.4 ms/token（瓶颈是 launch，不是带宽）。

新布局（v2）：segment s 覆盖 128 个连续 k（k = 128s .. 128s+127），打包成 64 字节：
    byte (64s + j) 的**低** 4 bit = W[128s + j]          (j = 0..63)
    byte (64s + j) 的**高** 4 bit = W[128s + 64 + j]
于是：
    lo tile 需要 x[128s .. 128s+63]   ← 连续
    hi tile 需要 x[128s+64 .. 128s+127] ← 连续
→ **x 完全不需要拆分**，两次连续 DataCopy 即可。

scale 侧的连带收益：gs=32，所以 lo tile 的 lane 0..31 = group 4s、lane 32..63 = group 4s+1；
hi tile 的 lane 0..31 = group 4s+2、lane 32..63 = group 4s+3
→ 每 segment 的掩码归约从 **8 次降到 4 次**（掩码只有两个：低32位 / 高32位）。

本脚本验证：
  1) 旧布局 → 新布局 的 nibble 重排是**无损**的（unpack 回去逐元素相等）
  2) 用新布局 + 连续 x 块算出来的 y，与用原权重（numpy 契约）算出的 y 一致
"""
import numpy as np

GS = 32


def unpack_old(p4, N, K):
    """旧契约：byte j 低 bit = W[2j]，高 bit = W[2j+1]"""
    b = p4.view(np.uint8)
    lo = ((b & 0x0F).astype(np.int8) ^ 8) - 8
    hi = (((b >> 4) & 0x0F).astype(np.int8) ^ 8) - 8
    w = np.empty((N, K), dtype=np.int8)
    w[:, 0::2] = lo
    w[:, 1::2] = hi
    return w


def to_v2(w, N, K):
    """[N][K] 有符号 nibble（-8..7）→ v2 打包 [N][K/2] uint8"""
    assert K % 128 == 0
    nb = K // 2
    out = np.zeros((N, nb), dtype=np.uint8)
    for s in range(K // 128):
        lo_k = np.arange(128 * s, 128 * s + 64)        # 低 4bit 覆盖
        hi_k = np.arange(128 * s + 64, 128 * s + 128)  # 高 4bit 覆盖
        lo_n = (w[:, lo_k].astype(np.int16) & 0x0F).astype(np.uint8)
        hi_n = (w[:, hi_k].astype(np.int16) & 0x0F).astype(np.uint8)
        out[:, 64 * s:64 * s + 64] = lo_n | (hi_n << 4)
    return out


def unpack_v2(p4, N, K):
    """v2 契约 → [N][K] 有符号 nibble"""
    b = p4.view(np.uint8)
    lo = ((b & 0x0F).astype(np.int8) ^ 8) - 8
    hi = (((b >> 4) & 0x0F).astype(np.int8) ^ 8) - 8
    w = np.empty((N, K), dtype=np.int8)
    for s in range(K // 128):
        w[:, 128 * s:128 * s + 64] = lo[:, 64 * s:64 * s + 64]
        w[:, 128 * s + 64:128 * s + 128] = hi[:, 64 * s:64 * s + 64]
    return w


def ref_v2_contiguous(p4v2, s4, x, N, K):
    """按 v2 布局 + **连续 x 块** 计算 y，模拟 kernel 的新内循环。"""
    b = p4v2.view(np.uint8)
    lo_all = ((b & 0x0F).astype(np.int8) ^ 8) - 8      # [N][K/2]
    hi_all = (((b >> 4) & 0x0F).astype(np.int8) ^ 8) - 8
    y = np.zeros(N, dtype=np.float64)
    for s in range(K // 128):
        lo = lo_all[:, 64 * s:64 * s + 64].astype(np.float64)      # lanes 0..63
        hi = hi_all[:, 64 * s:64 * s + 64].astype(np.float64)
        x_lo = x[128 * s:128 * s + 64]                             # 连续块 0
        x_hi = x[128 * s + 64:128 * s + 128]                       # 连续块 1
        # 掩码：lane 0..31 / 32..63（对应 group 4s / 4s+1，以及 4s+2 / 4s+3）
        for half in range(2):
            a, z = half * 32, half * 32 + 32
            gl = 4 * s + half
            gh = 4 * s + 2 + half
            y += (lo[:, a:z] * x_lo[a:z]).sum(1) * s4[:, gl].astype(np.float64)
            y += (hi[:, a:z] * x_hi[a:z]).sum(1) * s4[:, gh].astype(np.float64)
    return y


def main():
    rng = np.random.RandomState(0)
    for N, K in ((1280, 2560), (2560, 640)):
        w = rng.randint(-8, 8, size=(N, K)).astype(np.int8)      # 直接给 nibble 值
        s4 = (rng.rand(N, K // GS).astype(np.float16) * 0.01).astype(np.float16)
        x = rng.randn(K).astype(np.float32)

        # 旧布局（仅用于对照：把 w 按旧契约打包）
        p4_old = np.zeros((N, K // 2), dtype=np.uint8)
        p4_old = ((w[:, 0::2].astype(np.int16) & 0x0F) |
                  ((w[:, 1::2].astype(np.int16) & 0x0F) << 4)).astype(np.uint8)

        # 1) 无损性：v2 打包后再解包，必须逐元素相等
        p4v2 = to_v2(w, N, K)
        w_back = unpack_v2(p4v2, N, K)
        lossless = np.array_equal(w, w_back)

        # 2) 数值：v2 布局 + 连续 x 块  ==  原始权重 @ x
        ref = np.einsum("nk,k->n", w.astype(np.float64) *
                        np.repeat(s4.astype(np.float64), GS, axis=1), x.astype(np.float64))
        got = ref_v2_contiguous(p4v2, s4, x, N, K)
        d = np.abs(got - ref)
        rel = float(d.mean() / (np.abs(ref).mean() + 1e-12))
        size = p4_old.nbytes
        print(f"N={N} K={K}  v2 打包体积 {size} B（与旧布局相同：{p4v2.nbytes == size}）", flush=True)
        print(f"  无损重排        : {'PASS' if lossless else 'FAIL'}", flush=True)
        print(f"  连续 x 块数值对拍: rel_mean {rel*100:.6f}%  max_abs {d.max():.3e}  "
              f"-> {'PASS' if rel < 1e-12 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
