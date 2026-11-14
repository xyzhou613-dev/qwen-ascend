"""Replace T+1 strided conv copies with one overlapped-read view copy."""
import os
import torch
import verifier_state as state
original=state.save_conv_prefixes

def save(layer,seq,T):
    mode=os.environ.get('R161_CONV_MODE','packed')
    if mode=='legacy':return
    if mode=='loop':return original(layer,seq,T)
    assert mode=='packed'
    key=(layer.S.data_ptr(),T)
    if key not in state.conv_journals:
        state.conv_journals[key]=torch.empty((T+1,)+tuple(layer.conv.shape),device=seq.device,dtype=layer.conv.dtype)
    view=seq.unfold(1,3,1).permute(1,0,2).unsqueeze(1)
    state.conv_journals[key].copy_(view)

def install():state.save_conv_prefixes=save

@torch.no_grad()
def validate(layer):
    result=[]
    for T in (2,4):
        for amplitude in (.003,.03,.3,1.):
            seq=torch.randn((layer.conv.shape[1],T+3),device=layer.dev,dtype=layer.conv.dtype)*amplitude
            original(layer,seq,T)
            ref=state.conv_journals[(layer.S.data_ptr(),T)].cpu().clone()
            os.environ['R161_CONV_MODE']='packed';save(layer,seq,T)
            got=state.conv_journals[(layer.S.data_ptr(),T)].cpu()
            assert torch.equal(ref,got),('PACKED_CONV_JOURNAL_DIFF',T,amplitude)
            result.append(dict(T=T,amplitude=amplitude,bitwise_equal=True))
    return result
