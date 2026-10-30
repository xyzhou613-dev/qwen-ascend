"""步骤 (b)：单层前向数值对齐。

对 linear_attention / full_attention / PLE 层各做：
  1) (impl ) 我的实现 + **原始 bf16 权重**  vs  transformers 参考逐层实现 → 验证实现正确性
  2) (quant) 我的实现 + **W4/W8 反量化权重** vs  参考(bf16)                 → 量化误差
用法：python tests/t_b_layer.py <layer_idx>
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/root/ascend-dev/fnx")
from fnx import engine as E                       # noqa: E402
from fnx.load import QDIR, SRCDIR, load_layer      # noqa: E402

from safetensors import safe_open                  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa
from transformers.models.qwen4_exp.modeling_qwen4_exp import (                          # noqa
    Qwen4ExpTextDecoderLayer, Qwen4ExpTextRotaryEmbedding)

DEV = "npu:0"
EXPERT_KEYS = ("mlp.experts.gate_up_proj", "mlp.experts.down_proj")
FULL_TYPES = E.FULL_TYPES


class ExpBf16:
    """bf16 专家权重（原始 checkpoint），用于实现对齐。"""
    def __init__(self, w):
        self.w = w

    def gather_deq(self, idx):
        return self.w.index_select(0, idx)


def load_bf16_layer(L):
    wm = json.load(open(os.path.join(SRCDIR, "model.safetensors.index.json")))["weight_map"]
    pre = f"model.language_model.layers.{L}."
    names = [n for n in wm if n.startswith(pre) and "ngram_embedding" not in n]
    byf = {}
    for n in names:
        byf.setdefault(wm[n], []).append(n)
    sd = {}
    for f, ns in byf.items():
        with safe_open(os.path.join(SRCDIR, f), framework="pt", device="cpu") as fh:
            for n in ns:
                sd[n[len(pre):]] = fh.get_tensor(n)
    return sd


def wrap(Wd):
    out = {}
    for k, v in Wd.items():
        if k in EXPERT_KEYS:
            out[k] = ExpBf16(v)
        elif v.dim() == 2:
            out[k] = E.Raw(v)
        else:
            out[k] = v
    return out


def main(L):
    cfg = Qwen4ExpTextConfig(**json.load(open(os.path.join(SRCDIR, "config.json")))["text_config"])
    ltype = "full" if cfg.layer_types[L] in FULL_TYPES else ("ple" if L == E.PLE_LAYER else "linear")
    print(f"=== layer {L}  type={ltype} ({cfg.layer_types[L]}) ===", flush=True)

    sd = load_bf16_layer(L)
    ref = Qwen4ExpTextDecoderLayer(cfg, L)
    miss, unexp = ref.load_state_dict(sd, strict=False)
    print("  load_state_dict missing:", [m for m in miss if "ngram" not in m][:6], "unexpected:", list(unexp)[:6])
    ref = ref.to(DEV, dtype=torch.bfloat16).eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    rot = Qwen4ExpTextRotaryEmbedding(cfg).to(DEV)
    pid = torch.zeros(4, 1, 1, dtype=torch.long, device=DEV)
    xin = torch.zeros(1, 1, E.HCH, dtype=torch.bfloat16, device=DEV)
    cos, sin = rot(xin, pid[1:])
    mc, ms = E.rope_cs(torch.zeros((), dtype=torch.int64, device=DEV), E.inv_freq(DEV))
    print(f"  rope 对齐: cos maxdiff {(mc.float()-cos[:, -1, :].view(-1).float()).abs().max().item():.3e} "
          f"sin maxdiff {(ms.float()-sin[:, -1, :].view(-1).float()).abs().max().item():.3e}")

    mask = torch.zeros(1, 1, 1, 1, dtype=torch.bfloat16, device=DEV)
    cmask = torch.ones(1, 1, dtype=torch.bool, device=DEV)
    pos = torch.zeros((), dtype=torch.int64, device=DEV)

    def mk(W):
        lay = E.FnLayer(L, ltype, W, DEV)
        if ltype == "full":
            lay.kv = E.KVStore(2048, DEV, 1)
            lay.ki = 0
        return lay

    def rel(a, b):
        d = (a.float() - b.float()).abs()
        return d.mean().item() / b.float().abs().mean().clamp_min(1e-9).item(), d.max().item()

    Wb = wrap({k: v.to(DEV) for k, v in sd.items()})
    Wq = load_layer(os.path.join(QDIR, f"L{L:02d}.safetensors"), L, DEV)

    for tag, seed in (("A", 0), ("B", 7)):
        torch.manual_seed(seed)
        h = torch.randn(1, 1, E.HCH, dtype=torch.bfloat16, device=DEV) * 0.1
        ng = torch.randn(1, 1, E.H, dtype=torch.bfloat16, device=DEV) * 0.1 if ltype == "ple" else None
        with torch.no_grad():
            t = time.time()
            o_ref = ref(h.clone(), (cos, sin), attention_mask=mask, conv_mask=cmask, past_key_values=None)
            torch.npu.synchronize()
            t_ref = time.time() - t
            lay = mk(Wb)
            t = time.time()
            o_imp = lay.forward(h.clone(), pos, ng)
            torch.npu.synchronize()
            t_imp = time.time() - t
            lay = mk(Wq)
            t = time.time()
            o_q = lay.forward(h.clone(), pos, ng)
            torch.npu.synchronize()
            t_q = time.time() - t
        r1, m1 = rel(o_imp, o_ref)
        r2, m2 = rel(o_q, o_ref)
        r3, m3 = rel(o_q, o_imp)
        print(f"\n  [{tag}] ref {t_ref*1e3:7.2f}ms | impl {t_imp*1e3:6.2f}ms | quant {t_q*1e3:6.2f}ms")
        print(f"      [impl ] 我(bf16)  vs 参考(bf16) : rel {r1*100:7.3f}%  max_abs {m1:.4e}")
        print(f"      [quant] 我(W4W8)  vs 参考(bf16) : rel {r2*100:7.3f}%  max_abs {m2:.4e}")
        print(f"      [净量化] 我(W4W8) vs 我(bf16)    : rel {r3*100:7.3f}%  max_abs {m3:.4e}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 0)
