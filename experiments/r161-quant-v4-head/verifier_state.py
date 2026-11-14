"""O(T) speculative state storage; never copy a full long-context cache at runtime."""
import torch
from fnx import engine as E
conv_journals={}

def save_conv_prefixes(layer,seq,T):
    key=(layer.S.data_ptr(),T)
    if key not in conv_journals:
        conv_journals[key]=torch.empty((T+1,)+tuple(layer.conv.shape),device=seq.device,dtype=layer.conv.dtype)
    journal=conv_journals[key]
    for cut in range(T+1):journal[cut].copy_(seq[:,cut:cut+3].unsqueeze(0))

class PrefixState:
    def __init__(self,layer,pos,T):
        self.layer,self.pos,self.T=layer,pos,T
        self.linear=layer.ltype in ('linear','ple')
        if self.linear:return
        self.offsets=torch.arange(T,device=pos.device,dtype=torch.int64)
        self.positions=pos.reshape(-1)[:1]+self.offsets
        count=(T+6)//4
        self.group_offsets=torch.arange(count,device=pos.device,dtype=torch.int64)
        self.groups=(self.positions[0]//4+self.group_offsets).clamp_max(self.positions[-1]//4)
        self.kv_tensors=layer.kv.k+layer.kv.v
        self.old_kv=[torch.empty_like(x.index_select(2,self.positions)) for x in self.kv_tensors]
        self.old_raw=torch.empty_like(layer.index_k.index_select(0,self.positions))
        self.old_pool=torch.empty_like(layer.index_keys.index_select(0,self.groups))

    def begin(self):
        if self.linear:return
        self.positions.copy_(self.pos.reshape(-1)[:1]+self.offsets)
        self.groups.copy_((self.positions[0]//4+self.group_offsets).clamp_max(self.positions[-1]//4))
        for dst,src in zip(self.old_kv,self.kv_tensors):dst.copy_(src.index_select(2,self.positions))
        self.old_raw.copy_(self.layer.index_k.index_select(0,self.positions))
        self.old_pool.copy_(self.layer.index_keys.index_select(0,self.groups))

    def restore(self,cut):
        assert 0<=cut<=self.T
        if cut==self.T:return
        layer=self.layer
        if self.linear:
            import small_delta
            layer.S.copy_(small_delta.journals[(layer.S.data_ptr(),self.T)][cut])
            layer.conv.copy_(conv_journals[(layer.S.data_ptr(),self.T)][cut])
            return
        suffix=self.positions[cut:]
        for dst,src in zip(self.kv_tensors,self.old_kv):dst.index_copy_(2,suffix,src[:,:,cut:])
        layer.index_k.index_copy_(0,suffix,self.old_raw[cut:])
        # A rejected suffix may share a 4-row pool with retained tokens.
        # Restore original pools, then rebuild only groups touched by retained raw keys.
        layer.index_keys.index_copy_(0,self.groups,self.old_pool)
        if cut:
            kept=self.positions[:cut]
            layer._record_index_keys(layer.index_k.index_select(0,kept),kept)

    def bytes(self):
        if self.linear:
            import small_delta
            tensors=[small_delta.journals[(self.layer.S.data_ptr(),self.T)],conv_journals[(self.layer.S.data_ptr(),self.T)]]
        else:tensors=self.old_kv+[self.old_raw,self.old_pool]
        return sum(x.numel()*x.element_size() for x in tensors)

def capture_cpu(layer):
    out={'conv':layer.conv.detach().cpu().clone()}
    if layer.ltype in ('linear','ple'):out['S']=layer.S.detach().cpu().clone()
    else:
        out['kv']=[x.detach().cpu().clone() for x in layer.kv.k+layer.kv.v]
        out['raw']=layer.index_k.detach().cpu().clone()
        out['pool']=layer.index_keys.detach().cpu().clone()
    return out

def load_cpu(layer,state):
    layer.conv.copy_(state['conv'])
    if 'S' in state:layer.S.copy_(state['S'])
    else:
        for dst,src in zip(layer.kv.k+layer.kv.v,state['kv']):dst.copy_(src)
        layer.index_k.copy_(state['raw']);layer.index_keys.copy_(state['pool'])

def equal_cpu(a,b):
    if a.keys()!=b.keys():return False
    return all(all(torch.equal(x,y) for x,y in zip(a[k],b[k])) if isinstance(a[k],list) else torch.equal(a[k],b[k]) for k in a)

def expected_prefix(layer,pre,post,backup,cut):
    """Validation-only CPU reconstruction; these full copies are excluded from timing."""
    if layer.ltype in ('linear','ple'):
        import small_delta
        return {'S':small_delta.journals[(layer.S.data_ptr(),backup.T)][cut].cpu().clone(),
                'conv':conv_journals[(layer.S.data_ptr(),backup.T)][cut].cpu().clone()}
    positions=backup.positions.cpu();kept=positions[:cut]
    out={'conv':pre['conv'].clone(),'kv':[x.clone() for x in pre['kv']],
         'raw':pre['raw'].clone(),'pool':pre['pool'].clone()}
    for dst,src in zip(out['kv'],post['kv']):
        if cut:dst.index_copy_(2,kept,src.index_select(2,kept))
    if cut:
        out['raw'].index_copy_(0,kept,post['raw'].index_select(0,kept))
        if cut>1 and __import__('os').environ.get('FNX_INDEX_UNIQUE')=='1':
            gids=(kept[0]//4+torch.arange((cut+6)//4)).clamp_max(kept[-1]//4)
        else:gids=kept//4
        rows=(gids[:,None]*4+torch.arange(4)).reshape(-1)
        raw=out['raw'].index_select(0,rows).to(layer.dev).view(-1,4,128)
        pooled=raw.float().mean(1).to(layer.index_k.dtype)
        gd=gids.to(layer.dev);cos,sin=E.rope_cs(gd*4,layer.inv)
        keys=E.apply_rope(E.rms_norm(pooled,layer.index_kn),cos,sin).float().cpu()
        out['pool'].index_copy_(0,gids,keys)
    return out
