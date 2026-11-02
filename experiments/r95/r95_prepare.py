from pathlib import Path
import shutil
r=Path('/root/ascend-dev');s=r/'r95';s.mkdir(exist_ok=True)
shutil.copytree(r/'fnx/fnx',s/'fnx',dirs_exist_ok=True)
# Keep group sorting on the measured faster original path.
candidate=(r/'r94/fnx/engine.py').read_text()
start=candidate.index('            if os.environ.get("FNX_INDEX_BLOCK_SORT"')
end=candidate.index('            selected.append(',start)
original=(r/'fnx/fnx/engine.py').read_text()
a=original.index('            tokens = (chosen[:, :, None]')
b=original.index('            selected.append(',a)
candidate=candidate[:start]+original[a:b]+candidate[end:]
(s/'fnx/engine.py').write_text(candidate)
p=s/'fnx/model.py';text=p.read_text()
text=text.replace('def save_state(self, to_host=False, compact=None):','def save_state(self, to_host=False, compact=None, dual_slim=None):')
needle='        kv_copy = lambda t: copy(t[:, :, :n])'
assert text.count(needle)==1
text=text.replace(needle,needle+'''
        if dual_slim is None:
            dual_slim = os.environ.get("FNX_STATE_DUAL_SLIM", "0") == "1"
        def kv_snapshot(kv):
            count = kv.n if dual_slim and getattr(kv, "native_dual", False) else len(kv.k)
            return ([kv_copy(t) for t in kv.k[:count]], [kv_copy(t) for t in kv.v[:count]])''')
text=text.replace('"kv0": ([kv_copy(t) for t in self.kv0.k], [kv_copy(t) for t in self.kv0.v]),\n              "kv1": ([kv_copy(t) for t in self.kv1.k], [kv_copy(t) for t in self.kv1.v])}',
'''"kv0": kv_snapshot(self.kv0), "kv1": kv_snapshot(self.kv1)}
        if dual_slim:
            st["kv_format"] = "original-only-v1"''')
needle='        self.pos0.copy_(st["pos0"])'
assert text.count(needle)==1
text=text.replace(needle,'''        if st.get("kv_format") not in (None, "original-only-v1"):
            raise ValueError("unsupported snapshot KV format")
'''+needle)
needle='''            for t, s in zip(kv.v, sv):
                t[:, :, :n].copy_(s)
        torch.npu.synchronize()'''
assert text.count(needle)==1
text=text.replace(needle,'''            for t, s in zip(kv.v, sv):
                t[:, :, :n].copy_(s)
            if st.get("kv_format") == "original-only-v1" and getattr(kv, "native_dual", False):
                if len(sk) != kv.n or len(sv) != kv.n:
                    raise ValueError("invalid original-only dual-cache snapshot")
                from .snapshot import rebuild_packed
                rebuild_packed(kv, n)
        torch.npu.synchronize()''')
needle='''            st.update(self.save_state(to_host=st["pos0"].device.type == "cpu", compact=True))
            return'''
assert text.count(needle)==1
text=text.replace(needle,'''            replacement = self.save_state(to_host=st["pos0"].device.type == "cpu", compact=True,
                                          dual_slim=st.get("kv_format") == "original-only-v1")
            st.clear()
            st.update(replacement)
            return''')
p.write_text(text)
p=s/'fnx/server.py';text=p.read_text()
needle='            "native_dual": os.environ.get("FNX_NATIVE_DUAL", "0") == "1",'
assert text.count(needle)==1
text=text.replace(needle,needle+'''
            "index_unique": os.environ.get("FNX_INDEX_UNIQUE", "0") == "1",
            "state_dual_slim": os.environ.get("FNX_STATE_DUAL_SLIM", "0") == "1",
            "conversation_state_bytes": sum(model.state_bytes(st) for st in getattr(STATE.get("e"), "conv_slots", [])),''')
p.write_text(text)
shutil.copy2(r/'r95_snapshot.py',s/'fnx/snapshot.py')
for p in (s/'fnx').glob('*.py'):compile(p.read_text(encoding='utf-8-sig'),str(p),'exec')
env=(r/'r92/candidate.env').read_text()+'\nFNX_INDEX_UNIQUE=1\nFNX_STATE_DUAL_SLIM=1\n'
(s/'candidate.env').write_text(env)
for name in ('r51_payloads.py','r51_measure.py','r52_long_client.py'):
    shutil.copy2(r/'r92'/name,s/name)
print('R95_PREPARED',flush=True)
