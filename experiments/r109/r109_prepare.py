from pathlib import Path
import shutil
r=Path('/root/ascend-dev');s=r/'r109';s.mkdir(exist_ok=True)
shutil.copytree(r/'fnx/fnx',s/'fnx',dirs_exist_ok=True)
shutil.copy2(r/'r109_delta_native.py',s/'fnx/delta_native.py')
shutil.copy2(r/'r108/libfnx_delta.so',s/'fnx/libfnx_delta.so')
p=s/'fnx/engine.py';text=p.read_text()
start=text.index('        S = self.S * torch.exp(g)[..., None, None]')
end=text.index('        o = o.to(torch.bfloat16)',start)
block=text[start:end]
text=text[:start]+'''        if b == 1 and os.environ.get("FNX_DN_NATIVE", "0") == "1" and os.environ.get("FNX_DN_DECODE_BMM", "0") != "1":
            from .delta_native import step as native_delta_step
            o = native_delta_step(self.S, q, k, v, beta, torch.exp(g)[..., None, None])
        else:
'''+''.join('    '+line if line.strip() else line for line in block.splitlines(True))+text[end:]
p.write_text(text);compile(text,str(p),'exec')
p=s/'fnx/server.py';text=p.read_text();needle='            "delta_window": int(os.environ.get("FNX_DELTA_WINDOW", "0")),'
assert text.count(needle)==1
text=text.replace(needle,needle+'\n            "delta_native": os.environ.get("FNX_DN_NATIVE", "0") == "1" and os.environ.get("FNX_DN_DECODE_BMM", "0") != "1",')
p.write_text(text);compile(text,str(p),'exec')
(s/'candidate.env').write_text((r/'r106/candidate.env').read_text()+'\nFNX_DN_NATIVE=1\n')
for name in ('r51_payloads.py','r51_measure.py','r52_long_client.py'):shutil.copy2(r/'r106'/name,s/name)
print('R109_PREPARED',flush=True)
