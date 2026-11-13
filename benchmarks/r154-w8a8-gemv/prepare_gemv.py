"""CPU-only lossless repack of a real expert subset for a GEMV experiment."""
import json,time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file
P=Path(__file__).resolve().parent;torch.set_num_threads(2)
cfg=json.load(open(P/'production-selected-config.json'))
source=Path(cfg['FNX_W4A16_DIR']+'-em')/'L00.w4.safetensors'
(P/'private-artifacts').mkdir(exist_ok=True);records=[]
with safe_open(str(source),framework='pt',device='cpu') as f:
    for name in ('gate_up_proj','down_proj'):
        start=time.monotonic();key='mlp.experts.'+name
        w=f.get_tensor(key+'.w4');s=f.get_tensor(key+'.s8t');base=f.get_tensor(key+'.base')
        count=256;K=w.shape[1];N=w.shape[2]*8;G=K//64;GP=(G+7)//8*8
        packed=torch.empty(count,N,K//2,dtype=torch.uint8)
        for first in range(0,count,4):
            z=w[first:first+4]
            nibbles=torch.stack([(z >> shift)&15 for shift in range(0,32,4)],-1).to(torch.uint8)
            rows=nibbles.reshape(z.shape[0],K,N).transpose(1,2).contiguous()
            packed[first:first+4].copy_(rows[:,:,0::2] | (rows[:,:,1::2]<<4))
        scales=torch.zeros(count,N,GP,dtype=torch.bfloat16)
        sc=(s[:,:count,:].permute(1,0,2).to(torch.bfloat16)*base[:count,None,:]).transpose(1,2)
        scales[:,:,:G].copy_(sc)
        # Independent scalar bit checks against the source K-major int32 words.
        for expert in (0,17,255):
            for row,k in ((0,0),(31,63),(N-1,K-1),(N//2,64)):
                expected=(int(w[expert,k,row//8])>>((row%8)*4))&15
                actual=(int(packed[expert,row,k//2])>>((k%2)*4))&15
                assert expected==actual
        out=P/'private-artifacts'/f'L00-{name}-rowpack256.safetensors'
        save_file({'packed':packed,'scale':scales},str(out))
        records.append(dict(name=name,source=str(source),experts=count,K=K,N=N,G=G,scale_padded_G=GP,
            output_bytes=out.stat().st_size,elapsed_s=time.monotonic()-start,repack='lossless; no quantization change'))
        print('REPACK_DONE',records[-1],flush=True)
        del packed,scales,sc,w,s,base,nibbles,rows,z
(P/'gemv-repack-manifest.json').write_text(json.dumps(records,indent=2))
