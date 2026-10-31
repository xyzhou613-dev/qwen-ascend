"""Real-weight grouped MoE precision and captured throughput."""
import os
import torch
from fnx import engine as E
from fnx.load import load_layer,QDIR
from t_r50_layer_profile import measure


def main():
    torch.npu.set_device(0)
    os.environ['FNX_MOE_GROUP']='1'
    W=load_layer(os.path.join(QDIR,'L00.safetensors'),0,'npu:0')
    L=E.FnLayer(0,'linear',W,'npu:0'); del W
    for T in (8,32,64,128,512):
        h=torch.randn(T,1,E.H,dtype=torch.bfloat16,device='npu:0')*.02
        old=lambda:torch.cat([L._moe(h[t:t+1]) for t in range(T)],0)
        new=lambda:L._moe(h)
        ref=old(); got=new()
        d=(ref.float()-got.float()).abs()
        print('GMM_REAL',T,{'max':float(d.max()),'rel':float(d.mean()/(ref.float().abs().mean()+1e-9)),
                           'old_ms':measure(old),'new_ms':measure(new)},flush=True)


if __name__=='__main__':
    with torch.no_grad():main()
