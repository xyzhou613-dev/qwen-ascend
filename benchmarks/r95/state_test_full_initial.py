"""Snapshot format compatibility, exact dual reconstruction and slot reuse."""
import gc,json,os,time,statistics
from pathlib import Path
from types import SimpleNamespace
import torch,torch_npu
from fnx.model import FnModel
from fnx.engine import KVStore
from fnx.snapshot import rebuild_packed

def equal(a,b):
    if torch.is_tensor(a):
        aa=a.cpu();bb=b.cpu()
        if aa.dtype==torch.bfloat16:aa=aa.view(torch.int16);bb=bb.view(torch.int16)
        assert torch.equal(aa,bb),(a.shape,a.dtype)
    elif isinstance(a,dict):
        for k,v in a.items():equal(v,b[k])
    elif isinstance(a,(tuple,list)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b,(a,b)

@torch.no_grad()
def main():
    torch.npu.set_device(0);torch.manual_seed(9595)
    m=FnModel.__new__(FnModel);m.P=262144;m.ctx=[1,2,3]
    m.pos0=torch.tensor([m.P],device='npu:0');m.pos1=m.pos0.clone();m._host_pos=m.P
    m.kv0=KVStore(m.P,'npu:0',1);m.kv1=KVStore(m.P,'npu:0',1)
    m.layers0=[SimpleNamespace(conv=torch.randn(1,8,3,device='npu:0'),S=torch.randn(1,2,8,8,device='npu:0'),
        index_k=torch.randn(m.P,128,dtype=torch.bfloat16,device='npu:0'),
        index_keys=torch.randn(m.P//4,128,device='npu:0'),use_ple=False)]
    m.layers1=[]
    for kv in (m.kv0,m.kv1):
        for t in kv.k[:kv.n]+kv.v[:kv.n]:
            # Include NaNs, signed zeros and every BF16 bit pattern.
            t.copy_(torch.randint(-32768,32767,t.shape,dtype=torch.int16,device='npu:0').view(torch.bfloat16))
        rebuild_packed(kv,m.P)
    full=m.save_state(dual_slim=False);slim=m.save_state(dual_slim=True)
    saved=m.state_bytes(full);reduced=m.state_bytes(slim)
    assert saved-reduced==2*2*m.P*576*2
    timings=[]
    for n in (0,1,3,4,7,31,8192,m.P):
        m.load_state(full);m.pos0.fill_(n);m.pos1.fill_(n)
        for host in (False,True):
            old=m.save_state(to_host=host,compact=True,dual_slim=False)
            st=m.save_state(to_host=host,compact=True,dual_slim=True)
            for kv in (m.kv0,m.kv1):
                for t in kv.k+kv.v:t.zero_()
            begin=time.perf_counter();m.load_state(st);elapsed=(time.perf_counter()-begin)*1000
            actual=m.save_state(to_host=host,compact=True,dual_slim=False);equal(old,actual)
            m.save_into(st);m.load_state(FnModel.state_to_host(st));equal(old,m.save_state(to_host=host,compact=True,dual_slim=False))
            if n==m.P:timings.append({'host':host,'load_ms':elapsed})
            del st,old,actual;gc.collect()
        print('SNAPSHOT_EXACT',n,flush=True)
    # Reallocate a compact slot while retaining its explicit format, regardless
    # of current environment defaults; validate legacy full-format reuse too.
    m.pos0.fill_(3);m.pos1.fill_(3);st=m.save_state(compact=True,dual_slim=True)
    m.pos0.fill_(7);m.pos1.fill_(7);os.environ['FNX_STATE_DUAL_SLIM']='0';m.save_into(st)
    assert st['valid_len']==7 and st['kv_format']=='original-only-v1'
    m.load_state(st)
    m.load_state(full);m.save_into(full);m.load_state(full)
    equal(full,m.save_state(dual_slim=False))
    for kv in (m.kv0,m.kv1):
        kv.native_dual=False
    st=m.save_state(dual_slim=True);m.load_state(st)
    assert len(st['kv0'][0])==len(full['kv0'][0])
    result={'full_bytes':saved,'slim_bytes':reduced,'saved_bytes':saved-reduced,'load_timings':timings,
        'lengths':[0,1,3,4,7,31,8192,m.P],'bf16_all_bit_patterns':True,'legacy_host_slot_reuse':True}
    Path('/root/ascend-dev/r95/state-test.json').write_text(json.dumps(result,indent=2))
    print('R95_STATE_ACCEPT',json.dumps(result),flush=True)

if __name__=='__main__':main()
