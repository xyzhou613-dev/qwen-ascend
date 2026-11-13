"""Compare manual validated SmoothQuant wiring with the selected engine hook."""
import gc,hashlib,json,os,sys
from pathlib import Path
os.environ['FNX_CTX']='8192'
import torch,torch_npu
from fnx import engine as E
from fnx.load import load_layer,QDIR
from smooth_w8a8 import configure_layer
P=Path(__file__).resolve().parent;mode=sys.argv[1]
torch.npu.set_device(0);torch.set_num_threads(1);torch.manual_seed(154432)
torch.npu.set_per_process_memory_fraction(1/32,0)
def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
@torch.no_grad()
def main():
 records=[]
 for index,kind in [(0,'linear'),(3,'full'),(24,'linear'),(27,'full')]:
  os.environ['FNX_DENSE_W8A8']='1' if mode=='selected' else '0'
  W=load_layer(f'{QDIR}/L{index:02d}.safetensors',index,'npu:0');L=E.FnLayer(index,kind,W,'npu:0');del W
  if mode!='selected':L.w8a8_replacements=configure_layer(L,E.Raw)
  if kind=='full':
   L.kv=E.KVStore(8192,'npu:0',1);L.index_k=torch.zeros(8192,128,device='npu',dtype=torch.bfloat16);L.index_keys=torch.zeros(2048,128,device='npu',dtype=torch.float32)
  x=torch.randn(4,1,E.HCH,device='npu',dtype=torch.bfloat16)*.03;pos=torch.tensor([4096],device='npu',dtype=torch.int64)
  hashes=[]
  for i in range(4):
   pos.fill_(4096+i);out=L.forward(x[i:i+1],pos)
   hashes.append(dict(output=digest(out),conv=digest(L.conv),state=digest(L.S) if kind=='linear' else digest(L.kv.k[0])))
  records.append(dict(layer=index,replacements=L.w8a8_replacements,hashes=hashes))
  del L,x,pos,out;E._GAMMA_CACHE.clear();gc.collect();torch.npu.empty_cache()
 (P/f'loader-{mode}.json').write_text(json.dumps(dict(status='passed',mode=mode,engine_module=E.__file__,records=records),indent=2))
 print('LOADER_CHECK_DONE',mode,flush=True)
if __name__=='__main__':main()
