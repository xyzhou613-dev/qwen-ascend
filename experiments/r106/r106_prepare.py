from pathlib import Path
import shutil
r=Path('/root/ascend-dev');s=r/'r106';s.mkdir(exist_ok=True)
shutil.copytree(r/'fnx/fnx',s/'fnx',dirs_exist_ok=True)
p=s/'fnx/engine.py';text=p.read_text()
needle='''                    inverse = torch.empty_like(order).index_copy_(
                        0, order, torch.arange(order.numel(), device=x.device))'''
assert text.count(needle)==1
text=text.replace(needle,'''                    if b >= 4096 and order.numel() <= 16777216 and os.environ.get("FNX_MOE_INVERSE_SORT", "0") == "1":
                        # order is a permutation of [0,N). Its inverse is
                        # argsort(order); FP32 is exact for every index here.
                        # Large 1-D int64 index_copy is slow on this CANN build.
                        inverse = torch.argsort(order.float())
                    else:
                        inverse = torch.empty_like(order).index_copy_(
                            0, order, torch.arange(order.numel(), device=x.device))''')
p.write_text(text);compile(text,str(p),'exec')
p=s/'fnx/server.py';text=p.read_text();needle='            "moe_inverse": os.environ.get("FNX_MOE_INVERSE", "0") == "1",'
assert text.count(needle)==1
text=text.replace(needle,needle+'\n            "moe_inverse_sort": os.environ.get("FNX_MOE_INVERSE_SORT", "0") == "1",')
p.write_text(text);compile(text,str(p),'exec')
(s/'candidate.env').write_text((r/'r100/candidate.env').read_text()+'\nFNX_MOE_INVERSE_SORT=1\n')
for name in ('r51_payloads.py','r51_measure.py','r52_long_client.py'):shutil.copy2(r/'r100'/name,s/name)
test=(r/'r100_moe_test.py').read_text().replace('(128,512,8192)','(128,512,4096,8192)').replace('/root/ascend-dev/r100/moe-test.json','/root/ascend-dev/r106/moe-test.json').replace('R100_MOE_ACCEPT','R106_MOE_ACCEPT')
(s/'moe_test.py').write_text(test)
print('R106_PREPARED',flush=True)
