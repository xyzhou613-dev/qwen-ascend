"""Private diagnostic copies only; production model sources stay untouched."""
import hashlib,json,pathlib,shutil,sys
P=pathlib.Path(__file__).resolve().parent;LIVE=pathlib.Path('/root/ascend-dev/fnx/fnx')
R156=pathlib.Path('/models/fnx-diagnostics/r156-moe-kmajor-20261113')
assert not (P/'baseline').exists(),'refuse overwrite staged runtime'
for phase,source in [('baseline',LIVE),('smooth',pathlib.Path('/models/fnx-diagnostics/r154-w8a8-gemv-20261113/selected/fnx')),('combined',R156/'candidate/fnx')]:
 dest=P/phase/'fnx';shutil.copytree(source,dest)
 for name in ('kmajor_down.py','libfnx_w4_kmajor.so'):shutil.copyfile(R156/name,dest/name)
 shutil.copyfile(P/'audit.py',dest/'audit.py')
 f=dest/'engine.py';s=f.read_text();s+='\nfrom .audit import install as _install_r157_audit\n_install_r157_audit(__import__("sys").modules[__name__])\n';f.write_text(s)
 f=dest/'pipeline.py';s=f.read_text();start=s.index('    @torch.no_grad()\n    def step_pad(');end=s.index('\n    def step(',start)
 new=s[start:end].replace('def step_pad(self,tok,with_head=True):','def step_score(self,tok,target):').replace("self.last_token=self._rpc('step',bool(with_head))","return self._rpc('qa_score',int(target))")
 assert 'def step_score' in new and "'qa_score'" in new
 s=s[:end]+'\n'+new+s[end:];f.write_text(s)
 f=dest/'pipeline_worker.py';s=f.read_text();start=s.index("            elif op=='step':");end=s.index("            elif op=='sample':",start)
 new=s[start:end].replace("elif op=='step':","elif op=='qa_score':").replace('with_head,=args','target,=args\n                with_head=True')
 new+='''                lg=m.row_logits().float().view(-1)
                value=(float(-torch.log_softmax(lg,-1)[int(target)]),int(value))
'''
 s=s[:end]+new+s[end:]
 anchor="            if op=='peer_init':";assert s.count(anchor)==1
 s=s.replace(anchor,"            if op=='qa_diag':\n                from .audit import status\n                value=status()\n            elif op=='peer_init':",1);f.write_text(s)
 f=dest/'server.py';s=f.read_text();s=s.replace('"service_revision": "r152",','"service_revision": "r152", "diagnostic_variant": os.environ["R157_PHASE"], "dense_w8a8_enabled": os.environ.get("FNX_DENSE_W8A8")=="1", "kmajor_down_enabled": os.environ.get("FNX_KMAJOR_DOWN")=="1",',1)
 anchor='if __name__ == "__main__":';assert s.count(anchor)==1
 s=s.replace(anchor,'from qa_routes import install as _install_r157_routes\n_install_r157_routes(globals())\n\n'+anchor,1).replace('host="0.0.0.0"','host="127.0.0.1"');f.write_text(s)
 for file in dest.glob('*.py'):compile(file.read_bytes(),str(file),'exec')
manifest={phase:{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (P/phase/'fnx').iterdir() if f.is_file() and f.suffix in ('.py','.so')} for phase in ('baseline','smooth','combined')}
(P/'runtime-code-manifest.json').write_text(json.dumps(manifest,indent=2))
(P/'production-before-hashes.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in LIVE.iterdir() if f.is_file() and f.suffix in ('.py','.so')},indent=2))
shutil.copyfile('/root/ascend-dev/p275.sh',P/'p275-before.sh')
print('PRIVATE_RUNTIMES_PREPARED',list(manifest))
