"""步骤 (b) 诊断：逐子模块定位我的实现与参考的差异。"""
import json, os, sys, time
import torch
sys.path.insert(0, "/root/ascend-dev/fnx")
from fnx import engine as E
from fnx.load import SRCDIR
from tests.t_b_layer import load_bf16_layer, wrap, DEV
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer, Qwen4ExpTextRotaryEmbedding)

L = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cfg = Qwen4ExpTextConfig(**json.load(open(os.path.join(SRCDIR, "config.json")))["text_config"])
ltype = "full" if cfg.layer_types[L] == "full_attention" else ("ple" if L == E.PLE_LAYER else "linear")
sd = load_bf16_layer(L)
ref = Qwen4ExpTextDecoderLayer(cfg, L)
ref.load_state_dict(sd, strict=False)
ref = ref.to(DEV, dtype=torch.bfloat16).eval()
Wb = wrap({k: v.to(DEV) for k, v in sd.items()})
lay = E.FnLayer(L, ltype, Wb, DEV)
if ltype == "full":
    lay.kv = E.KVStore(2048, DEV, 1); lay.ki = 0
rot = Qwen4ExpTextRotaryEmbedding(cfg).to(DEV)
cos, sin = rot(torch.zeros(1, 1, E.HCH, dtype=torch.bfloat16, device=DEV),
               torch.zeros(3, 1, 1, dtype=torch.long, device=DEV))
cos1, sin1 = cos[:, -1:, :].view(1, 64), sin[:, -1:, :].view(1, 64)
cmask = torch.ones(1, 1, dtype=torch.bool, device=DEV)
pos = torch.zeros((), dtype=torch.int64, device=DEV)

def rel(a, b):
    d = (a.float() - b.float()).abs()
    return d.mean().item() / b.float().abs().mean().clamp_min(1e-9).item(), d.max().item()

torch.manual_seed(0)
h = torch.randn(1, 1, E.HCH, dtype=torch.bfloat16, device=DEV) * 0.1
print(f"layer {L} type={ltype}  |h|mean={h.float().abs().mean():.4f}")

with torch.no_grad():
    # 1) gated residual
    m_ref, hyp_ref, inj_ref = ref.attn_hyper_connection(h)
    m_me, inj_me = E.gated_residual(h, lay.a_hc_norm, lay.a_down, lay.a_up, lay.a_inj)
    print(f"  1) HC mixed  rel {rel(m_me, m_ref)[0]*100:6.3f}%  max {rel(m_me, m_ref)[1]:.3e}"
          f"   inj rel {rel(inj_me, inj_ref)[0]*100:6.3f}%  |m|={m_ref.float().abs().mean():.4f}")

    # 2) attention branch
    if ltype in ("linear", "ple"):
        a_ref = ref.linear_attn(m_ref, cache_params=None, attention_mask=cmask)
        a_me = lay._deltanet(m_ref)
    else:
        a_ref, _ = ref.self_attn(m_ref, (cos1, sin1), attention_mask=torch.zeros(1,1,1,1, dtype=torch.bfloat16, device=DEV),
                                 past_key_values=None)
        a_me = lay._qsa(m_ref, pos)
    print(f"  2) attn      rel {rel(a_me, a_ref)[0]*100:6.3f}%  max {rel(a_me, a_ref)[1]:.3e}"
          f"  |a|={a_ref.float().abs().mean():.4f}")

    h2_ref = hyp_ref + (a_ref.unsqueeze(-2)*inj_ref.unsqueeze(-1)).flatten(-2)
    h2_me = E.inject(h, inj_me, a_me)
    print(f"     h after inj rel {rel(h2_me, h2_ref)[0]*100:6.3f}%  max {rel(h2_me, h2_ref)[1]:.3e}")

    # 3) MoE
    m2_ref, hyp2_ref, inj2_ref = ref.mlp_hyper_connection(h2_ref)
    moe_ref = ref.mlp(m2_ref)
    logits_ref = ref.mlp.gate(m2_ref.view(1, E.H))[0]
    with torch.no_grad():
        moe_me = lay._moe(m2_me if False else m2_ref)   # 同一输入，隔离 MoE 本身
        logits_me = torch.nn.functional.linear(m2_ref.view(1, E.H), lay.gate_w)
    print(f"  3) router logits rel {rel(logits_me, logits_ref)[0]*100:6.3f}%  "
          f"max {rel(logits_me, logits_ref)[1]:.3e}")
    top_ref = torch.topk(torch.softmax(logits_ref.float(), -1), E.TOPK, -1).indices[0].tolist()
    top_me = torch.topk(torch.softmax(logits_me.float(), -1), E.TOPK, -1).indices[0].tolist()
    print(f"     top10 一致率 {len(set(top_ref)&set(top_me))/E.TOPK*100:.0f}%  ref={top_ref}  me={top_me}")
    print(f"  4) MoE out   rel {rel(moe_me, moe_ref)[0]*100:6.3f}%  max {rel(moe_me, moe_ref)[1]:.3e}"
          f"  |moe|={moe_ref.float().abs().mean():.4f}")

    # 5) 手工重算 ref MoE（用参考的路由与专家，验证我的 bmm 与线性层等价）
    sh = torch.nn.functional.linear(
        torch.nn.functional.silu(torch.nn.functional.linear(m2_ref.view(1,E.H), ref.mlp.shared_expert.gate_proj.weight))
        * torch.nn.functional.linear(m2_ref.view(1,E.H), ref.mlp.shared_expert.up_proj.weight),
        ref.mlp.shared_expert.down_proj.weight)
    sh = torch.sigmoid(torch.nn.functional.linear(m2_ref.view(1,E.H), ref.mlp.shared_expert_gate.weight)) * sh
    print(f"  5) shared expert rel {rel(lay._moe.__self__ if False else sh, moe_ref)[0]*100:6.3f}%  (仅参考量级)")
