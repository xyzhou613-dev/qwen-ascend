"""Lossless expert-major int4 layout for grouped prefill; retain original packs."""
import os,time
import torch
from safetensors.torch import load_file,save_file


def main():
    source=os.environ.get('FNX_W4A16_DIR','/models/Qwen3.8-Flash-Next-LQP-w4a16')
    dest=source+'-em'
    os.makedirs(dest,exist_ok=True)
    for i in range(48):
        target=f'{dest}/L{i:02d}.w4.safetensors'
        if os.path.exists(target):continue
        t=time.perf_counter()
        data=load_file(f'{source}/L{i:02d}.w4.safetensors')
        out={k:(v.permute(1,0,2).contiguous() if k.endswith('.w4') else v) for k,v in data.items()}
        # Byte-for-byte transpose round trip is the format acceptance gate.
        for k,v in data.items():
            if k.endswith('.w4'):assert torch.equal(v,out[k].permute(1,0,2))
        save_file(out,target+'.tmp')
        os.replace(target+'.tmp',target)
        del data,out
        print('REPACK',i,round(time.perf_counter()-t,2),flush=True)


if __name__=='__main__':main()
