#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void norm_ordered_bf16(
 GM_ADDR input,GM_ADDR table,GM_ADDR output,int rows,int token_stride,int scale_query) {
 GlobalTensor<bfloat16_t> gx,gy;GlobalTensor<uint16_t> gt;
 gx.SetGlobalBuffer((__gm__ bfloat16_t*)input);gy.SetGlobalBuffer((__gm__ bfloat16_t*)output);gt.SetGlobalBuffer((__gm__ uint16_t*)table);
 TPipe pipe;TBuf<TPosition::VECCALC> xb,fb,pb,rb,wb,sb;
 pipe.InitBuffer(xb,256);pipe.InitBuffer(fb,512);pipe.InitBuffer(pb,512);
 pipe.InitBuffer(rb,256);pipe.InitBuffer(wb,512);pipe.InitBuffer(sb,32);
 auto x=xb.Get<bfloat16_t>();auto f=fb.Get<float>();auto p=pb.Get<float>();
 auto r=rb.Get<bfloat16_t>();auto work=wb.Get<float>();auto sum=sb.Get<float>();
 for(int row=GetBlockIdx();row<rows;row+=GetBlockNum()) {
  int64_t offset=static_cast<int64_t>(row/16)*token_stride+(row%16)*128;
  DataCopy(x,gx[offset],128);SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  Cast(f,x,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
  Mul(p,f,f,128);PipeBarrier<PIPE_V>();
  Cast(r,p,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();
  Cast(p,r,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
  // Match CANN BF16 R=128: accumulate the two 64-lane chunks before horizontal reduction.
  Add(p,p,p[64],64);PipeBarrier<PIPE_V>();ReduceSum(sum,p,work,64);PipeBarrier<PIPE_V>();
  Cast(r,sum,RoundMode::CAST_RINT,1);
  SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
  union {uint32_t bits;float value;} u;u.bits=static_cast<uint32_t>(gt.GetValue(r.ReinterpretCast<uint16_t>().GetValue(0)))<<16;
  Duplicate(p,u.value,128);PipeBarrier<PIPE_V>();Mul(f,f,p,128);PipeBarrier<PIPE_V>();
  Cast(r,f,RoundMode::CAST_RINT,128);
  if(scale_query && row<16) {
   // Preserve the normalization's BF16 rounding before q / sqrt(128).
   PipeBarrier<PIPE_V>();Cast(f,r,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
   Muls(f,f,0.08838834764831845f,128);PipeBarrier<PIPE_V>();Cast(r,f,RoundMode::CAST_RINT,128);
  }
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(gy[static_cast<int64_t>(row)*128],r,128);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" int fnx_norm_ordered(void*x,void*t,void*y,int rows,int stride,int scale_query,void*stream) {
 if(rows<16||rows%16||stride<2048||(scale_query&&rows!=32))return -1;int blocks=rows<40?rows:40;
 norm_ordered_bf16<<<blocks,nullptr,(aclrtStream)stream>>>((GM_ADDR)x,(GM_ADDR)t,(GM_ADDR)y,rows,stride,scale_query);return 0;
}
