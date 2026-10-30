"""步骤 (e)：8 层 + lm_head + tokenizer 的真实端到端（文本生成），实测 TPOT。"""
import sys, time, torch
sys.path.insert(0, "/root/ascend-dev/fnx")
from fnx import engine as E
from fnx.model import FnModel

PROMPT = "用一句话解释什么是线性注意力。"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 8

m = FnModel(P=2048, lpc=N, dev0="npu:0", dev1="npu:0", nlayers=N, with_head=True, verbose=True)
t = time.time(); m.capture(); print(f"capture {time.time()-t:.1f}s", flush=True)
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("/models/Qwen3.8-Flash-Next")
ids = tok(tok.apply_chat_template([{"role":"user","content":PROMPT}], tokenize=False,
                                  add_generation_prompt=True), return_tensors="pt").input_ids[0].tolist()
print(f"prompt tokens {len(ids)}", flush=True)

m.reset()
for x in ids[:-1]: m.step(int(x), with_head=False)
nxt = int(ids[-1]); out=[]
t = time.time()
for _ in range(16):
    m.step(nxt, with_head=True)
    nxt = int(m.logits.float().argmax(-1).item()); out.append(nxt)
t_gen = time.time()-t
print("输出:", repr(tok.decode(out, skip_special_tokens=True)), flush=True)

m.reset()
for x in ids[:-1]: m.step(int(x), with_head=False)
for _ in range(3): m.step(ids[-1], with_head=True)
torch.npu.synchronize(); t=time.perf_counter()
for _ in range(20): m.step(ids[-1], with_head=True)
torch.npu.synchronize(); tpot=(time.perf_counter()-t)/20*1e3
print(f"[{N} 层] TPOT {tpot:.3f} ms/token → 外推 48 层 {tpot/N*48:.2f} ms", flush=True)
print(f"含 host 同步的生成 TPOT {t_gen/16*1e3:.3f} ms/token")
print(f"HBM npu0 {torch.npu.memory_allocated(0)/2**30:.2f} GiB")
print("OK")