"""Generate readable supplemental sources, preserving every frozen file."""
import hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
def replace(s,old,new):
    assert s.count(old)==1,old
    return s.replace(old,new)
s=(P/'batch_ops.py').read_text()
s=replace(s,'import torch_npu','import torch_npu\nimport write_nd as ND')
s=replace(s,'kc.scatter_(2,where,k[:,:,None,:]);vc.scatter_(2,where,v[:,:,None,:])','ND.kv(kc,k,pos);ND.kv(vc,v,pos)')
s=replace(s,'self.index_k.scatter_(1,pos[:,None,None].expand(b,1,128),raw[:,None,:])','ND.index(self.index_k,raw,pos)')
s=replace(s,'self.index_keys.scatter_(1,gid[:,None,None].expand(b,1,128),pooled[:,None,:].float())','ND.index(self.index_keys,pooled.float(),gid)')
(P/'batch_ops_nd.py').write_text(s)
s=(P/'bench.py').read_text()
s=replace(s,'import batch_ops as O','import batch_ops_nd as O')
s=replace(s,'for rep in range(3):','for rep in range(3 if target==8000 else 1):')
s=replace(s,'results=[];eos=tokenizer.eos_token_id','results=[];eos={tokenizer.eos_token_id,248044,248046}')
s=replace(s,'if next_ids[i]==eos:','if next_ids[i] in eos:')
s=replace(s,'out[-1]==eos','out[-1] in eos')
s=replace(s,'r=dict(batch=B,rep=rep,requested_context=target,','r=dict(batch=B,rep=rep,prefill_chunk=E._CHUNK,cache_writer="scatter_nd_update_inplace",requested_context=target,')
(P/'bench_nd.py').write_text(s)
print('ND_SUPPLEMENTAL_SOURCES_GENERATED')
