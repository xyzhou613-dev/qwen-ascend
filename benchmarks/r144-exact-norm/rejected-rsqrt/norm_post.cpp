#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
// Keep Torch's BF16 squared values and ReduceSum. Round after epsilon and
// Rsqrt exactly where the original eager expression materializes BF16.
extern "C" __global__ __aicore__ void norm_post(GM_ADDR input,GM_ADDR sums,GM_ADDR output,int rows,int token_stride) {
 GlobalTensor<bfloat16_t> gx,gy;GlobalTensor<uint16_t> gs;
 gx.SetGlobalBuffer((__gm__ bfloat16_t*)input);gy.SetGlobalBuffer((__gm__ bfloat16_t*)output);gs.SetGlobalBuffer((__gm__ uint16_t*)sums);
 TPipe pipe;TBuf<TPosition::VECCALC> xb,fb,db,rb;
 pipe.InitBuffer(xb,256);pipe.InitBuffer(fb,512);pipe.InitBuffer(db,512);pipe.InitBuffer(rb,256);
 auto x=xb.Get<bfloat16_t>();auto f=fb.Get<float>();auto d=db.Get<float>();auto r=rb.Get<bfloat16_t>();
 for(int row=GetBlockIdx();row<rows;row+=GetBlockNum()) {
  int64_t offset=static_cast<int64_t>(row/16)*token_stride+(row%16)*128;
  DataCopy(x,gx[offset],128);
  union {uint32_t bits;float value;} u;u.bits=static_cast<uint32_t>(gs.GetValue(row))<<16;
  Duplicate(d,u.value,128);PipeBarrier<PIPE_V>();Adds(d,d,1e-6f,128);PipeBarrier<PIPE_V>();
  Cast(r,d,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();Cast(d,r,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
  Rsqrt(d,d,128);PipeBarrier<PIPE_V>();Cast(r,d,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();Cast(d,r,RoundMode::CAST_NONE,128);
  SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);Cast(f,x,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
  Mul(f,f,d,128);PipeBarrier<PIPE_V>();Cast(r,f,RoundMode::CAST_RINT,128);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(gy[static_cast<int64_t>(row)*128],r,128);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" int fnx_norm_post(void*x,void*s,void*y,int rows,int stride,void*stream) {
 if(rows<16||rows%16||stride<2048)return -1;
 int blocks=rows<40?rows:40;
 norm_post<<<blocks,nullptr,(aclrtStream)stream>>>((GM_ADDR)x,(GM_ADDR)s,(GM_ADDR)y,rows,stride);return 0;
}
// A finite BF16 sum has only 32768 nonnegative bit patterns. Build the
// complete 65536-entry table using the ORIGINAL CANN add/Rsqrt expression.
// This avoids substituting the approximate vector Rsqrt instruction.
extern "C" __global__ __aicore__ void norm_lookup(GM_ADDR input,GM_ADDR sums,GM_ADDR table,GM_ADDR output,int rows,int token_stride) {
 GlobalTensor<bfloat16_t> gx,gy;GlobalTensor<uint16_t> gs,gt;
 gx.SetGlobalBuffer((__gm__ bfloat16_t*)input);gy.SetGlobalBuffer((__gm__ bfloat16_t*)output);gs.SetGlobalBuffer((__gm__ uint16_t*)sums);gt.SetGlobalBuffer((__gm__ uint16_t*)table);
 TPipe pipe;TBuf<TPosition::VECCALC> xb,fb,db,rb;
 pipe.InitBuffer(xb,256);pipe.InitBuffer(fb,512);pipe.InitBuffer(db,512);pipe.InitBuffer(rb,256);
 auto x=xb.Get<bfloat16_t>();auto f=fb.Get<float>();auto d=db.Get<float>();auto r=rb.Get<bfloat16_t>();
 for(int row=GetBlockIdx();row<rows;row+=GetBlockNum()) {
  int64_t offset=static_cast<int64_t>(row/16)*token_stride+(row%16)*128;
  DataCopy(x,gx[offset],128);
  union {uint32_t bits;float value;} u;u.bits=static_cast<uint32_t>(gt.GetValue(gs.GetValue(row)))<<16;
  Duplicate(d,u.value,128);
  SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);Cast(f,x,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
  Mul(f,f,d,128);PipeBarrier<PIPE_V>();Cast(r,f,RoundMode::CAST_RINT,128);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(gy[static_cast<int64_t>(row)*128],r,128);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" int fnx_norm_lookup(void*x,void*s,void*t,void*y,int rows,int stride,void*stream) {
 if(rows<16||rows%16||stride<2048)return -1;int blocks=rows<40?rows:40;
 norm_lookup<<<blocks,nullptr,(aclrtStream)stream>>>((GM_ADDR)x,(GM_ADDR)s,(GM_ADDR)t,(GM_ADDR)y,rows,stride);return 0;
}
