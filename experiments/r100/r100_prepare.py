from pathlib import Path
import shutil
r=Path('/root/ascend-dev');s=r/'r100';s.mkdir(exist_ok=True)
shutil.copytree(r/'fnx/fnx',s/'fnx',dirs_exist_ok=True)
p=s/'fnx/engine.py';text=p.read_text();needle='                restored = torch.empty_like(yd).index_copy_(0, order, yd)'
assert text.count(needle)==1
text=text.replace(needle,'''                if b >= 512 and os.environ.get("FNX_MOE_INVERSE", "0") == "1":
                    # Invert only the small row permutation, then gather the
                    # large expert output. Keep token/expert order, BF16 multiply
                    # and sum exactly as in the original scatter path.
                    inverse = torch.empty_like(order).index_copy_(
                        0, order, torch.arange(order.numel(), device=x.device))
                    restored = yd.index_select(0, inverse)
                else:
                    restored = torch.empty_like(yd).index_copy_(0, order, yd)''')
p.write_text(text);compile(text,str(p),'exec')
p=s/'fnx/server.py';text=p.read_text();needle='            "index_unique": os.environ.get("FNX_INDEX_UNIQUE", "0") == "1",'
assert text.count(needle)==1
text=text.replace(needle,needle+'\n            "moe_inverse": os.environ.get("FNX_MOE_INVERSE", "0") == "1",')
p.write_text(text);compile(text,str(p),'exec')
(s/'candidate.env').write_text((r/'r95/candidate.env').read_text()+'\nFNX_MOE_INVERSE=1\n')
for name in ('r51_payloads.py','r51_measure.py','r52_long_client.py'):shutil.copy2(r/'r95'/name,s/name)
print('R100_PREPARED',flush=True)
