import numpy as np
np.long=np.int64
np.ulong=np.uint64
import torch,torch_npu
import json
from pathlib import Path
torch.npu.config.allow_internal_format=True;torch.npu.set_device(0);torch.set_num_threads(1);torch.npu.set_per_process_memory_fraction(1/128,0)
x=torch.randn(1,2560,device='npu',dtype=torch.bfloat16);w=torch.randn(512,2560,device='npu',dtype=torch.bfloat16)*.01
scale=w.float().abs().amax(1).clamp_min(1e-12)/127;q=(w.float()/scale[:,None]).round().clamp(-127,127).to(torch.int8)
xq,xs=torch_npu.npu_dynamic_quant(x)
for layout in ('transposed','nd','nz'):
 try:
  a=q.t() if layout=='transposed' else q.t().contiguous()
  if layout=='nz':a=torch_npu.npu_format_cast(a,29)
  y=torch_npu.npu_quant_matmul(xq,a,scale,pertoken_scale=xs,output_dtype=torch.bfloat16);torch.npu.synchronize();ref=torch.nn.functional.linear(x,w);print('QUANT_SMOKE',layout,y.shape,float((y.float()-ref.float()).norm()/ref.float().norm()),flush=True)
 except Exception as e:print('QUANT_LAYOUT_UNSUPPORTED',layout,type(e).__name__,str(e),flush=True)
