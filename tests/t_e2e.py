"""★ 端到端参考对拍：同一段 4 个 token、**同样截断到 2 层**，比较逐位置 logits。

为什么必须做这个：单层对拍（t_b_layer.py）只喂一个随机 h、只看一层，它覆盖不到
  - 残差流入口（`emb.repeat(hc)`）
  - 层与层之间的串接
  - `hyper_connection_mixer` + `lm_head` 这一对出口
而实测整机 top1≈0.5%、mean_nll≈ln(V)，单层却 PASS —— 差异只可能在这些没被覆盖的地方。
本脚本让"参考实现"和"我们的实现"跑**完全相同的截断**（2 层），任何差异都必然出在我们的代码里。

用法（容器内，需先停服腾显存）：
  cd /root/ascend-dev/fnx && /root/ascend-dev/venv/bin/python tests/t_e2e.py
"""
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from fnx import engine as E
from fnx.load import QDIR, SRCDIR
from fnx.model import FnModel

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import (                          # noqa: E402
    Qwen4ExpTextDecoderLayer, Qwen4ExpTextGatedResidual, Qwen4ExpTextRotaryEmbedding)

DEV = "npu:0"
NL = 2
T = 4


def _wm():
    return json.load(open(os.path.join(SRCDIR, "model.safetensors.index.json")))["weight_map"]


def load_keys(names):
    """按 index 分文件加载指定权重键（去掉 model.language_model. 前缀）。"""
    wm = _wm()
    pre = "model.language_model."
    byf = {}
    for n in names:
        byf.setdefault(wm[n], []).append(n)
    sd = {}
    for f, ns in byf.items():
        with safe_open(os.path.join(SRCDIR, f), framework="pt", device="cpu") as fh:
            for n in ns:
                key = n[len(pre):] if n.startswith(pre) else n
                sd[key] = fh.get_tensor(n)
    return sd


class RefMini(nn.Module):
    """严格照抄 Qwen4ExpTextModel.forward 的骨架（截断到 NL 层）。"""

    def __init__(self, cfg, nlayers=NL, disable_ple=True):
        super().__init__()
        self.cfg = cfg
        self.hc = cfg.hc_count
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id)
        self.layers = nn.ModuleList([Qwen4ExpTextDecoderLayer(cfg, i) for i in range(nlayers)])
        self.rot = Qwen4ExpTextRotaryEmbedding(cfg)
        # ★ 属性名必须与 checkpoint 键名一致（hyper_connection_mixer.*）：
        # 早先叫 self.mixer，权重没被 load 进去（missing=4），参考侧 mixer 是随机的，
        # 导致整机对拍假报 404% —— 而同一位置 T=1 的严格对拍其实是 3.7%。
        self.hyper_connection_mixer = Qwen4ExpTextGatedResidual(cfg, use_combine=False)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if disable_ple:
            for L in self.layers:
                if getattr(L, "ple", None) is not None:
                    L.ple = None

    @torch.no_grad()
    def forward(self, ids):
        Tn = ids.shape[1]
        h = self.embed(ids)                                  # [1,T,H]
        pid = torch.arange(Tn, device=ids.device).view(1, 1, Tn)
        pe = self.rot(h, pid)
        h = h.repeat(1, 1, self.hc)                          # ★ 入口：与参考一致
        # ★ 掩码必须用 transformers 自己的工具生成：手写加性掩码会被 QSA 的 indexer
        # 当成 bool 掩码误判（t_e2e.py 第一版 409% 的假警报就是这么来的）。
        from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
        mk = dict(config=self.cfg, inputs_embeds=h, attention_mask=None,
                  past_key_values=None, position_ids=pid, allow_is_causal_skip=False)
        am = create_causal_mask(**mk)
        cmask = create_recurrent_attention_mask(**mk)
        for L in self.layers:
            h = L(h, pe, attention_mask=am, conv_mask=cmask, past_key_values=None)
        h = self.hyper_connection_mixer(h)
        return self.lm_head(h)                               # [1,T,V]


def rel(a, b):
    d = (a.float() - b.float()).abs()
    return float(d.mean() / b.float().abs().mean().clamp_min(1e-9)), float(d.max())


def main():
    cfg = Qwen4ExpTextConfig(**json.load(open(os.path.join(SRCDIR, "config.json")))["text_config"])
    # ★ 参考侧不要构造 PLE 模块：Qwen4ExpTextDecoderLayer 在 PLE 层会初始化 51 GB 的 ngram 表，
    # 实测直接把进程挂死（>240 s 无进展）。我们两侧都关掉 PLE，保证比较的是同一条计算路径。
    cfg.ple_layer_ids = []
    print(f"=== 端到端对拍：{NL} 层 / {T} token / hc_count={cfg.hc_count} "
          f"hidden={cfg.hidden_size} vocab={cfg.vocab_size} ===", flush=True)
    print(f"    PLE_LAYER={getattr(E, 'PLE_LAYER', '?')}  layer_types[:4]={cfg.layer_types[:4]} "
          f"(参考侧 ple_layer_ids 已清空)", flush=True)

    # ---------- 参考实现 ----------
    print("    [1/4] 构造参考模型...", flush=True)
    ref = RefMini(cfg).to(DEV, dtype=torch.bfloat16).eval()
    print("    [2/4] 读 bf16 权重...", flush=True)
    names = ["model.language_model.embed_tokens.weight", "lm_head.weight"]
    for i in range(NL):
        names += [n for n in _wm() if n.startswith(f"model.language_model.layers.{i}.")
                  and "ngram_embedding" not in n]
    names += [n for n in _wm() if n.startswith("model.language_model.hyper_connection_mixer.")]
    sd = load_keys(names)
    miss, unexp = ref.load_state_dict(sd, strict=False)
    del sd
    print(f"    参考 load: missing={len([m for m in miss if 'ngram' not in m])} "
          f"unexpected={len(unexp)}", flush=True)
    for p in ref.parameters():
        p.requires_grad_(False)

    # ---------- 我们的实现（QP 权重） ----------
    print("    [3/4] 构造我们的模型（QP 权重）...", flush=True)
    m = FnModel(P=2048, lpc=NL, dev0=DEV, dev1="npu:1", nlayers=NL,
                with_head=True, with_ngram=False, verbose=False)
    for L in m.layers0 + m.layers1:
        if L.use_ple:
            L.use_ple = False              # 与参考侧一致地关掉 PLE
            print(f"    [注意] 我们的 L{L.idx} 是 PLE 层，已与参考一致地关掉 PLE")

    # ---------- 同一段 token ----------
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/models/Qwen3.8-Flash-Next")
    ids = tok("昇腾处理器面向大规模深度学习训练与推理场景", return_tensors="pt").input_ids[:, :T]
    print(f"    ids={ids.tolist()}  ->  {tok.decode(ids[0].tolist())!r}", flush=True)

    print("    [4/4] 前向对拍...", flush=True)
    with torch.no_grad():
        lg_ref = ref(ids.to(DEV))[0].float()                 # [T,V]
        m.reset()
        lg_our = []
        for i in range(ids.shape[1]):
            m.step(int(ids[0, i]), with_head=True)
            lg_our.append(m.logits.float().view(-1).clone())
        lg_our = torch.stack(lg_our)                         # [T,V]

    print("\n  pos | 参考 |logits| | 我们 |logits| | rel      | max_abs   | argmax 参考/我们")
    for i in range(T):
        r, mx = rel(lg_our[i], lg_ref[i])
        ar = int(lg_ref[i].argmax())
        ao = int(lg_our[i].argmax())
        print(f"   {i}  | {lg_ref[i].norm().item():10.4e} | {lg_our[i].norm().item():10.4e} "
              f"| {r*100:7.3f}% | {mx:.4e} | {ar} / {ao}"
              f"{'  <-- 同' if ar == ao else ''}")
    r_all, mx_all = rel(lg_our.reshape(-1), lg_ref.reshape(-1))
    verdict = "PASS(实现一致)" if r_all < 5e-2 else "FAIL(实现有差异，问题在这里)"
    print(f"\n  整体: rel {r_all*100:.3f}%  max_abs {mx_all:.4e}  -> {verdict}")
    print(f"  参考 top1 解码: {[tok.decode([int(x)]) for x in lg_ref.argmax(-1)]}")
    print(f"  我们 top1 解码: {[tok.decode([int(x)]) for x in lg_our.argmax(-1)]}")


if __name__ == "__main__":
    main()
