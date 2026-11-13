"""Stream real r152 hidden states to collect per-projection input maxima."""
import gc,hashlib,json,os,time
from pathlib import Path
import torch,torch_npu
from safetensors.torch import save_file
import sweep_suite as S
from fnx import engine as E
from fnx.load import load_layer,load_nonlayer,QDIR,SRCDIR
from fnx.model import FnModel
P=Path(__file__).resolve().parent

@torch.no_grad()
def main():
    cases=json.load(open(P/'calibration-inputs.json'))['cases']
    emb,unused,mx=load_nonlayer('npu:0','npu:0',with_lmhead=False)
    dummy=FnModel.__new__(FnModel);dummy.emb_q,dummy.emb_s=emb
    hidden=[dummy.embed_rows(c['ids'][:-1]).view(-1,1,E.H).repeat(1,1,E.HC) for c in cases]
    del emb,unused,mx,dummy
    cfg=json.load(open(Path(SRCDIR)/'config.json'))['text_config'];manifest=[]
    (P/'private-calibration').mkdir(exist_ok=True)
    for index in range(48):
        S.guard();E._GAMMA_CACHE.clear();os.environ['FNX_DENSE_W8A8']='0'
        ltype='full' if cfg['layer_types'][index] in E.FULL_TYPES else 'linear'
        W=load_layer(f'{QDIR}/L{index:02d}.safetensors',index,'npu:0');L=E.FnLayer(index,ltype,W,'npu:0');del W
        if ltype=='full':
            L.kv=E.KVStore(8192,'npu:0',1);L.short_qsa=True
            L.index_k=torch.zeros(8192,128,device='npu',dtype=torch.bfloat16)
            L.index_keys=torch.zeros(2048,128,device='npu',dtype=torch.float32)
        names=('w_qkv','w_out') if ltype=='linear' else ('w_q','w_o')
        weights={id(getattr(L,n)):n for n in names}
        maxima={n:torch.zeros(getattr(L,n).w.shape[1],device='npu',dtype=torch.float32) for n in names}
        original=E.lin
        def observed(x,w):
            name=weights.get(id(w))
            if name is not None:
                value=x.reshape(-1,x.shape[-1]).float().abs().amax(0)
                maxima[name].copy_(torch.maximum(maxima[name],value))
            return original(x,w)
        E.lin=observed
        try:hidden=S.run_documents(L,hidden)
        finally:E.lin=original
        result={k:v.cpu() for k,v in maxima.items()};file=P/'private-calibration'/f'L{index:02d}.safetensors'
        save_file(result,str(file))
        manifest.append(dict(layer=index,file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest(),channels={k:v.numel() for k,v in result.items()}))
        print('CALIBRATED',index,flush=True)
        del L,weights,maxima,observed;E._GAMMA_CACHE.clear();gc.collect();torch.npu.empty_cache()
    (P/'calibration-manifest.json').write_text(json.dumps(dict(layers=manifest,tokens=512,source='Separate frozen public/repository slices; no live user activations'),indent=2))

if __name__=='__main__':main()
