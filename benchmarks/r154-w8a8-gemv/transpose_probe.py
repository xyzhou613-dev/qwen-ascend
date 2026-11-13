"""Test exact packed-W4 layout conversion, including a full-layer round trip."""
import ctypes,json,statistics,time,urllib.request
from pathlib import Path
import torch,torch_npu
from safetensors import safe_open
P=Path(__file__).resolve().parent;torch.npu.set_device(0);torch.set_num_threads(1)
torch.npu.set_per_process_memory_fraction(1/32,0)
lib=ctypes.CDLL(str(P/'libfnx_w4_transpose.so'))
lib.fnx_w4_transpose.argtypes=[ctypes.c_void_p]*2+[ctypes.c_int]*3+[ctypes.c_void_p];lib.fnx_w4_transpose.restype=ctypes.c_int
def transpose(src,dst,rows,cols):
 assert src.dtype==torch.uint8 and dst.dtype==torch.uint8 and src.is_contiguous() and dst.is_contiguous()
 assert src.shape==(dst.shape[0],rows,cols//2) and dst.shape==(src.shape[0],cols,rows//2)
 rc=lib.fnx_w4_transpose(src.data_ptr(),dst.data_ptr(),src.shape[0],rows,cols,torch.npu.current_stream().npu_stream);assert rc==0,rc

@torch.no_grad()
def main():
 torch.manual_seed(15499);records=[]
 for experts,rows,cols in [(1,128,128),(3,256,512),(2,640,2560)]:
  q=torch.randint(-8,8,(experts,rows,cols),dtype=torch.int8)
  def pack(a):return ((a[:,:,::2].to(torch.int16)&15)|((a[:,:,1::2].to(torch.int16)&15)<<4)).to(torch.uint8).contiguous()
  src=pack(q).npu();dst=torch.empty((experts,cols,rows//2),device='npu',dtype=torch.uint8)
  back=torch.empty_like(src);transpose(src,dst,rows,cols);transpose(dst,back,cols,rows)
  assert torch.equal(dst.cpu(),pack(q.transpose(1,2).contiguous())) and torch.equal(src.cpu(),back.cpu())
  records.append(dict(experts=experts,rows=rows,cols=cols,exact=True))
  del src,dst,back,q;torch.npu.empty_cache()
 cfg=json.load(open(P/'production-selected-config.json'));file=Path(cfg['FNX_W4A16_DIR']+'-em')/'L00.w4.safetensors'
 with safe_open(str(file),framework='pt',device='cpu') as f:orig=f.get_tensor('mlp.experts.down_proj.w4').view(torch.uint8).reshape(512,640,1280)
 src=orig.npu();dst=torch.empty((512,2560,320),device='npu',dtype=torch.uint8);back=torch.empty_like(src)
 transpose(src,dst,640,2560);transpose(dst,back,2560,640)
 assert torch.equal(back.cpu(),orig)
 with safe_open(str(P/'private-artifacts/L00-down_proj-rowpack256.safetensors'),framework='pt',device='cpu') as f:
  assert torch.equal(dst[:256].cpu(),f.get_tensor('packed'))
 times={};stream=torch.npu.Stream()
 for label,a,b,rows,cols in [('to_row',src,dst,640,2560),('to_kmajor',dst,back,2560,640)]:
  with torch.npu.stream(stream):
   g=torch.npu.NPUGraph()
   with torch.npu.graph(g,stream=stream):
    for _ in range(8):transpose(a,b,rows,cols)
   stream.synchronize();values=[]
   for _ in range(8):
    h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5));assert h['service_revision']=='r152' and h['worker_alive']
    ae=torch.npu.Event(enable_timing=True);be=torch.npu.Event(enable_timing=True);ae.record(stream);g.replay();be.record(stream);be.synchronize();values.append(ae.elapsed_time(be)/8)
   g.reset();times[label]=values
 # Test phase changes with stable weight/scale pointers and one reusable scratch.
 with safe_open(str(file),framework='pt',device='cpu') as f:
  sc=(f.get_tensor('mlp.experts.down_proj.s8t').permute(1,0,2).to(torch.bfloat16)
      *f.get_tensor('mlp.experts.down_proj.base')[:,None,:]).contiguous()
 scbuf=torch.empty(512*2560*16,device='npu',dtype=torch.bfloat16)
 oldsc=scbuf[:sc.numel()].view_as(sc);oldsc.copy_(sc)
 rowsc=scbuf.view(512,2560,16);weight_ptr=src.data_ptr();scale_ptr=scbuf.data_ptr()
 def to_row():
  transpose(src,dst,640,2560);src.copy_(dst.reshape_as(src))
  temp=oldsc.transpose(1,2).contiguous();rowsc.zero_();rowsc[:,:,:10].copy_(temp)
 def to_original():
  transpose(src.reshape(512,2560,320),dst.reshape_as(src),2560,640);src.copy_(dst.reshape_as(src))
  temp=rowsc[:,:,:10].transpose(1,2).contiguous();oldsc.copy_(temp)
 to_row()
 with safe_open(str(P/'private-artifacts/L00-down_proj-rowpack256.safetensors'),framework='pt',device='cpu') as f:
  assert torch.equal(rowsc[:256].cpu(),f.get_tensor('scale'))
 to_original();assert torch.equal(src.cpu(),orig) and torch.equal(oldsc.cpu(),sc)
 assert src.data_ptr()==weight_ptr and scbuf.data_ptr()==scale_ptr
 cycle_times=[]
 with torch.npu.stream(stream):
  cycle=torch.npu.NPUGraph()
  with torch.npu.graph(cycle,stream=stream):
   for _ in range(4):to_row();to_original()
  stream.synchronize()
  for _ in range(8):
   ae=torch.npu.Event(enable_timing=True);be=torch.npu.Event(enable_timing=True)
   ae.record(stream);cycle.replay();be.record(stream);be.synchronize();cycle_times.append(ae.elapsed_time(be)/4)
  cycle.reset()
 assert torch.equal(src.cpu(),orig) and torch.equal(oldsc.cpu(),sc)
 result=dict(status='passed',synthetic=records,full_experts=512,full_roundtrip_exact=True,cpu_rowpack_exact=True,
     times_ms=times,median_ms={k:statistics.median(v) for k,v in times.items()},
     scratch_bytes=dst.numel(),phase_cycle_including_copies_and_scales_ms=cycle_times,
     median_phase_cycle_ms=statistics.median(cycle_times),weight_and_scale_pointers_stable=True,
     scale_extra_capacity_bytes=scbuf.numel()*2-sc.numel()*2,
     scope='Layout converter only; model phase lifecycle not implemented.',exclusive=False)
 (P/'transpose.json').write_text(json.dumps(result,indent=2));print('TRANSPOSE_RESULT',result['median_ms'],flush=True)
if __name__=='__main__':main()
