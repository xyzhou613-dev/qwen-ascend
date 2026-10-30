"""验证 engine.py 的 ExpQ 补丁：FNX_MOE_KERNEL=1 时能否正确建出 qv2 / s4t，
且这两份张量喂给 kernel 的结果 == ExpQ.gather_deq + bmm 的 torch 结果。

只建 ExpQ（不建整层），显存只多一份 10 专家的量，安全。
用法：FNX_MOE_KERNEL=1 python tests/t_expq_patch.py
"""
import os

import numpy as np
import torch
import torch_npu  # noqa: F401
from safetensors import safe_open

from fnx.engine import ExpQ
from fnx.w4gemv_op import W4GemvOp

QP = "/models/Qwen3.8-Flash-Next-QP"
GS = 32
nE = 10


def main():
    on = os.environ.get("FNX_MOE_KERNEL") == "1"
    print(f"[expq] FNX_MOE_KERNEL={'1' if on else '0(未开)'}", flush=True)
    if not on:
        print("[expq] 开关未开，本测试需要 FNX_MOE_KERNEL=1")
        return 1

    with safe_open(os.path.join(QP, "L00.packed.safetensors"), framework="pt") as f:
        for want in ("gate_up_proj", "down_proj"):
            kp = [k for k in f.keys() if want in k and k.endswith(".p4")][0]
            ks = [k for k in f.keys() if want in k and k.endswith(".s4")][0]
            p4 = f.get_slice(kp)[0:nE].numpy()
            s4 = f.get_slice(ks)[0:nE].numpy()
            N, Kh = p4.shape[1], p4.shape[2] * 2

            # 先把打包数据解成 int8 容器（服务 FNX_PACK=0 用的就是这个形态）
            bb = p4.view(np.uint8)
            lo = ((bb & 0x0F).astype(np.int8) ^ 8) - 8
            hi = (((bb >> 4) & 0x0F).astype(np.int8) ^ 8) - 8
            wint = np.empty((nE, N, Kh), dtype=np.int8)
            wint[..., 0::2] = lo
            wint[..., 1::2] = hi
            # 用 ExpQ 建（__init__ 在开关打开时会生成 qv2/s4t）
            eq = ExpQ(torch.from_numpy(wint).npu(), torch.from_numpy(s4).npu(), GS, packed=False)
            assert eq.qv2 is not None and eq.s4t is not None, "qv2/s4t 没建出来！"
            assert eq.q is None, "开关打开时 self.q 应已释放（否则显存翻倍）"
            print("[expq] self.q 已释放 -> 显存不会翻倍", flush=True)
            print(f"[expq] {want}: qv2 {tuple(eq.qv2.shape)} {eq.qv2.dtype}  "
                  f"s4t {tuple(eq.s4t.shape)} {eq.s4t.dtype}", flush=True)

            x = (np.random.RandomState(0).randn(Kh).astype(np.float32) * 0.3)
            # torch 路径（ExpQ 自己的 gather_deq + bmm，bf16）
            # torch 参考：直接用 int8 容器 + scale 复刻 bf16 语义（不依赖 ExpQ 内部状态）
            wq = (torch.from_numpy(wint).npu().to(torch.bfloat16).view(nE, N, Kh // GS, GS)
                  * torch.from_numpy(s4).npu().to(torch.bfloat16).unsqueeze(-1)).reshape(nE, N, Kh)
            xb = torch.from_numpy(x).npu().to(torch.bfloat16).expand(nE, Kh).unsqueeze(1)
            y_t = torch.bmm(xb, wq.transpose(1, 2)).squeeze(1).float()
            del wq

            # kernel 路径（用 ExpQ 建好的 qv2/s4t）
            op = W4GemvOp()
            xd = torch.from_numpy(np.ascontiguousarray(np.repeat(x[None, :], nE, 0))).npu()
            y_k = op.run(eq.qv2, eq.s4t, xd, nE, N, Kh, lay=2).float()

            d = (y_k - y_t).abs()
            rel = float(d.mean() / (y_t.abs().mean() + 1e-12))
            print(f"[expq] {want}: kernel(ExpQ.qv2/s4t) vs torch(int8容器+scale, bf16): "
                  f"rel_mean {rel*100:.3f}%  max {float(d.max()):.4f}  "
                  f"-> {'PASS' if rel < 5e-3 else 'FAIL'}", flush=True)
            del eq, y_k, y_t, xd
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
