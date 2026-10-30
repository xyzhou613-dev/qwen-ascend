"""★ head 精度 A/B：int8 量化 lm_head（现状） vs 原始 bf16 lm_head。

动机：输出里 'MoE'→'Mo'、'Qwen'→'Q'（掉一个 token），分词器已洗清（`MoE`=[24797,36] 往返正确），
      所以怀疑点是 lm_head 的 int8 分组量化把稀有 token 压掉了。这里直接换 head 量 NLL。

口径与 `/fnx/loss` 一致：teacher-forcing，`mean_nll` / `top1` / `median_rank`。
额外做定点探针：给 "Mo" 之后，`E`(36)、`wen`(16451) 的 NLL 与排名。
"""
import os
import time

import torch
from safetensors import safe_open

from fnx import engine as E
from fnx.model import FnModel

LPC = int(os.environ.get("PR_LPC", "44"))
NL = int(os.environ.get("PR_NL", "46"))
SRCDIR = os.environ.get("FNX_SRCDIR", "/models/Qwen3.8-Flash-Next")
HEAD_SHARD = os.path.join(SRCDIR, "model-00131-of-00131.safetensors")
EVAL = "/root/ascend-dev/eval_text.txt"
CAP = 200


@torch.no_grad()
def nll_run(m, ids, tag):
    m.reset()
    m.id = "ab"
    tot, hit, ranks = 0.0, 0, []
    t = time.perf_counter()
    for i in range(len(ids) - 1):
        m.step(int(ids[i]), with_head=True)
        lg = m.logits.float().view(-1)
        lp = torch.log_softmax(lg, dim=-1)
        tgt = ids[i + 1]
        tot += float(-lp[tgt].item())
        am = int(lg.argmax().item())
        hit += int(am == tgt)
        ranks.append(int((lg > lg[tgt]).sum().item()))
    dt = time.perf_counter() - t
    n = len(ids) - 1
    med = sorted(ranks)[len(ranks) // 2]
    print(f"  [{tag:9s}] n={n} mean_nll={tot/n:.4f} top1={hit/n*100:.2f}% "
          f"median_rank={med}  {dt/n*1e3:.1f} ms/token", flush=True)
    return tot / n, hit / n, med


@torch.no_grad()
def probe(m, text, tag, watch):
    """定点探针：逐 token teacher-forcing，打印指定 token 的 NLL 与排名。"""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SRCDIR)
    ids = tok(text, return_tensors="pt").input_ids[0].tolist()
    m.reset()
    m.id = "pr"
    out = []
    for i in range(len(ids) - 1):
        m.step(int(ids[i]), with_head=True)
        tgt = ids[i + 1]
        lg = m.logits.float().view(-1)
        lp = torch.log_softmax(lg, dim=-1)
        if tgt in watch:
            out.append((i, tgt, tok.decode([tgt]), float(-lp[tgt].item()),
                        int((lg > lg[tgt]).sum().item()), int(lg.argmax().item()),
                        tok.decode([int(lg.argmax())])))
    print(f"  [{tag}] {text!r}")
    for i, tgt, s, nll, rank, am, ams in out:
        print(f"      pos{i:2d} 目标 {tgt}({s!r}) NLL={nll:6.3f} rank={rank:5d} "
              f"argmax={am}({ams!r})", flush=True)
    return out


def main():
    m = FnModel(P=2048, lpc=LPC, dev0="npu:0", dev1="npu:1", nlayers=NL,
                with_head=True, with_ngram=False, verbose=False)
    m.capture()
    print(f"  装配完成 dev0={len(m.layers0)} dev1={len(m.layers1)} "
          f"H={E.H} vocab={m.logits.shape[-1]}", flush=True)
    print(f"  显存 alloc={torch.npu.memory_allocated(0)/2**30:.2f} GiB "
          f"reserved={torch.npu.memory_reserved(0)/2**30:.2f} GiB", flush=True)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SRCDIR)
    txt = open(EVAL, encoding="utf-8").read()
    ids = tok(txt, return_tensors="pt").input_ids[0].tolist()[:CAP + 1]
    print(f"\n=== 口径：eval_text 前 {len(ids)-1} token ===", flush=True)

    a = nll_run(m, ids, "int8-head")
    pa = probe(m, "混合专家（MoE）模型是一种稀疏架构。通义千问（Qwen）由阿里开发。",
               "int8-head", {36, 16451, 48})

    print("\n  换 bf16 head（原始权重 model-00131）...", flush=True)
    t = time.time()
    with safe_open(HEAD_SHARD, framework="pt", device="cpu") as f:
        w = f.get_tensor("lm_head.weight")
        print(f"    原始 lm_head: shape={tuple(w.shape)} dtype={w.dtype} "
              f"{w.numel()*w.element_size()/2**30:.2f} GiB", flush=True)
        wd = w.to("npu:0")
    del w
    old = m.lm_head
    m.lm_head = wd
    del old
    torch.npu.empty_cache()
    print(f"    装载 {time.time()-t:.1f}s  显存 alloc={torch.npu.memory_allocated(0)/2**30:.2f} "
          f"reserved={torch.npu.memory_reserved(0)/2**30:.2f} GiB", flush=True)

    b = nll_run(m, ids, "bf16-head")
    pb = probe(m, "混合专家（MoE）模型是一种稀疏架构。通义千问（Qwen）由阿里开发。",
               "bf16-head", {36, 16451, 48})

    print("\n=== 结论 ===")
    print(f"  mean_nll  {a[0]:.4f} -> {b[0]:.4f}  ({b[0]-a[0]:+.4f})")
    print(f"  top1      {a[1]*100:.2f}% -> {b[1]*100:.2f}%")
    print(f"  median_rank {a[2]} -> {b[2]}")
    ra = {i: r for i, _, _, _, r, _, _ in pa}
    rb = {i: r for i, _, _, _, r, _, _ in pb}
    for i in sorted(set(ra) | set(rb)):
        print(f"  探针 pos{i}: rank {ra.get(i)} -> {rb.get(i)}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
