import os,time
import torch
from fnx.model import FnModel

start=time.perf_counter()
P=int(os.environ['FNX_CTX'])
m=FnModel(P=P,lpc=int(os.environ['FNX_LPC']),dev0='npu:0',dev1='npu:1',nlayers=48,with_head=True,with_ngram=False,verbose=False)
m.capture();m.capture_chunk()
print('READY',P,'seconds',round(time.perf_counter()-start,2),'alloc0',torch.npu.memory_allocated(0)/2**30,'reserved0',torch.npu.memory_reserved(0)/2**30,flush=True)
