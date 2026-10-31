"""t_chunk_bisect: 定位分块 prefill 的数值偏差来源（第 46 轮）。

做法：同一段 prompt，先逐 token 跑（基准），再用 forward_chunk 以 T=1/2/4/8 跑，
每跑完就比对**状态张量**：
  * DeltaNet 的 S（第一个 linear 层）
  * conv 滑窗
  * KV cache（第一个 full 层）
这样能直接看出是递推/卷积错、还是注意力/KV 写错、还是纯 GEMM 批量舍入。
"""
import torch

TEXT_IDS = [18643, 11111, 22222, 33333, 44444, 55555, 66666, 77777]


def main():
    from fnx import engine as E
    from fnx.model import FnModel

    m = FnModel(P=2048, lpc=45, dev0="npu:0", dev1="npu:1", nlayers=48,
                with_head=True, with_ngram=False, verbose=False)
    lin_layers = [L for L in m.layers0 if L.ltype in ("linear", "ple")]
    full_layers = [L for L in m.layers0 if L.ltype == "full"]
    L0, F0 = lin_layers[0], full_layers[0]
    print(f"linear 层 {len(lin_layers)} 个（用 L{L0.idx}），full 层 {len(full_layers)} 个"
          f"（用 L{F0.idx}）", flush=True)

    def snapshot():
        return {
            "S": L0.S.clone(),
            "conv": L0.conv.clone(),
            "k": F0.kv.k[F0.ki].clone(),
            "v": F0.kv.v[F0.ki].clone(),
            "pos": m.pos0.clone(),
        }

    def cmp(tag, a, b):
        out = []
        for k in ("S", "conv", "k", "v"):
            d = (b[k].float() - a[k].float()).abs()
            sc = a[k].float().abs().mean().item() + 1e-9
            out.append(f"{k}: rel {d.mean().item()/sc:.2e} max {d.max().item():.2e}")
        print(f"  {tag:12s} " + "  |  ".join(out) + f"  pos {int(b['pos'][0])}", flush=True)

    # 基准：逐 token
    m.reset()
    for x in TEXT_IDS:
        m.step_pad(x, with_head=False)
    base = snapshot()
    print("\n=== 状态对比（相对基准）===", flush=True)

    for T in (1, 2, 4, 8):
        m.reset()
        E._CHUNK = T                      # 临时改块大小
        try:
            i = 0
            while i + T <= len(TEXT_IDS):
                m.prefill_chunk(TEXT_IDS[i:i + T])
                i += T
            for x in TEXT_IDS[i:]:
                m.step_pad(x, with_head=False)
            got = snapshot()
            cmp(f"chunk T={T}", base, got)
        except Exception as e:  # noqa: BLE001
            print(f"  chunk T={T} 失败 {type(e).__name__}: {str(e)[:160]}", flush=True)
        finally:
            E._CHUNK = 8

    # 再看 token 级影响：前 4 个 token 就比
    print("\n=== 只跑前 4 个 token（看偏差是否一开始就有）===", flush=True)
    m.reset()
    for x in TEXT_IDS[:4]:
        m.step_pad(x, with_head=False)
    base4 = snapshot()
    for T in (1, 4):
        m.reset()
        E._CHUNK = T
        try:
            i = 0
            while i + T <= 4:
                m.prefill_chunk(TEXT_IDS[i:i + T])
                i += T
            got = snapshot()
            cmp(f"T={T} (4 tok)", base4, got)
        finally:
            E._CHUNK = 8


if __name__ == "__main__":
    with torch.no_grad():
        main()
