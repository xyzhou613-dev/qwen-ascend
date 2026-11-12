#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;

// Input squares and both 16-row reductions stay in the original CANN ops.
// Only the two lookup/multiply epilogues and the rounded query scale combine.
extern "C" __global__ __aicore__ void norm_pair_post(
 GM_ADDR input,GM_ADDR qsum,GM_ADDR ksum,GM_ADDR table,GM_ADDR output,int scale_query) {
 GlobalTensor<bfloat16_t> gx,gy;GlobalTensor<uint16_t> gq,gk,gt;
 gx.SetGlobalBuffer((__gm__ bfloat16_t*)input);gy.SetGlobalBuffer((__gm__ bfloat16_t*)output);
 gq.SetGlobalBuffer((__gm__ uint16_t*)qsum);gk.SetGlobalBuffer((__gm__ uint16_t*)ksum);gt.SetGlobalBuffer((__gm__ uint16_t*)table);
 TPipe pipe;TBuf<TPosition::VECCALC> xb,fb,db,rb;
 pipe.InitBuffer(xb,256);pipe.InitBuffer(fb,512);pipe.InitBuffer(db,512);pipe.InitBuffer(rb,256);
 auto x=xb.Get<bfloat16_t>();auto f=fb.Get<float>();auto d=db.Get<float>();auto r=rb.Get<bfloat16_t>();
 const int row=GetBlockIdx();DataCopy(x,gx[row*128],128);
 uint16_t bits=row<16?gq.GetValue(row):gk.GetValue(row-16);
 union {uint32_t bits;float value;} u;u.bits=static_cast<uint32_t>(gt.GetValue(bits))<<16;
 Duplicate(d,u.value,128);SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
 Cast(f,x,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();Mul(f,f,d,128);PipeBarrier<PIPE_V>();
 Cast(r,f,RoundMode::CAST_RINT,128);
 if(scale_query && row<16) {
  PipeBarrier<PIPE_V>();Cast(f,r,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
  Muls(f,f,0.08838834764831845f,128);PipeBarrier<PIPE_V>();Cast(r,f,RoundMode::CAST_RINT,128);
 }
 SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(gy[row*128],r,128);
}
extern "C" int fnx_norm_pair(void*x,void*q,void*k,void*t,void*y,int scale_query,void*stream) {
 norm_pair_post<<<32,nullptr,(aclrtStream)stream>>>((GM_ADDR)x,(GM_ADDR)q,(GM_ADDR)k,(GM_ADDR)t,(GM_ADDR)y,scale_query);return 0;
}

// Research-only exports. No activation in production calls these kernels.
extern "C" __global__ __aicore__ void norm_sum_trace(
 GM_ADDR input,GM_ADDR square_out,GM_ADDR float_sum_out,GM_ADDR bf16_sum_out,int mode) {
 GlobalTensor<bfloat16_t> gx,gp,gb;GlobalTensor<float> gf;
 gx.SetGlobalBuffer((__gm__ bfloat16_t*)input);gp.SetGlobalBuffer((__gm__ bfloat16_t*)square_out);
 gb.SetGlobalBuffer((__gm__ bfloat16_t*)bf16_sum_out);gf.SetGlobalBuffer((__gm__ float*)float_sum_out);
 TPipe pipe;TBuf<TPosition::VECCALC> xb,fb,pb,rb,wb,sb;
 pipe.InitBuffer(xb,256);pipe.InitBuffer(fb,512);pipe.InitBuffer(pb,512);
 pipe.InitBuffer(rb,256);pipe.InitBuffer(wb,512);pipe.InitBuffer(sb,32);
 auto x=xb.Get<bfloat16_t>();auto f=fb.Get<float>();auto p=pb.Get<float>();auto r=rb.Get<bfloat16_t>();auto work=wb.Get<float>();auto sum=sb.Get<float>();
 int row=GetBlockIdx();DataCopy(x,gx[row*128],128);SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
 Cast(f,x,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();Mul(p,f,f,128);PipeBarrier<PIPE_V>();
 Cast(r,p,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();Cast(p,r,RoundMode::CAST_NONE,128);
 SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(gp[row*128],r,128);
 SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 if(mode==1){Add(p,p,p[64],64);PipeBarrier<PIPE_V>();ReduceSum(sum,p,work,64);}
 else ReduceSum(sum,p,work,128);
 SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(gf[row*8],sum,8);
 SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);Cast(r,sum,RoundMode::CAST_RINT,1);
 SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(gb[row*16],r,16);
}
extern "C" int fnx_norm_sum_trace(void*x,void*p,void*f,void*b,int mode,void*stream) {
 if(mode<0||mode>1)return -1;
 norm_sum_trace<<<16,nullptr,(aclrtStream)stream>>>((GM_ADDR)x,(GM_ADDR)p,(GM_ADDR)f,(GM_ADDR)b,mode);return 0;
}
extern "C" __global__ __aicore__ void norm_scale_trace(GM_ADDR input,GM_ADDR output) {
 GlobalTensor<bfloat16_t> gx,gy;gx.SetGlobalBuffer((__gm__ bfloat16_t*)input);gy.SetGlobalBuffer((__gm__ bfloat16_t*)output);
 TPipe pipe;TBuf<TPosition::VECCALC> xb,fb;pipe.InitBuffer(xb,256);pipe.InitBuffer(fb,512);
 auto x=xb.Get<bfloat16_t>();auto f=fb.Get<float>();
 for(int begin=GetBlockIdx()*128;begin<65536;begin+=GetBlockNum()*128) {
  DataCopy(x,gx[begin],128);SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  Cast(f,x,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();Muls(f,f,0.08838834764831845f,128);PipeBarrier<PIPE_V>();Cast(x,f,RoundMode::CAST_RINT,128);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(gy[begin],x,128);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" int fnx_norm_scale_trace(void*x,void*y,void*stream) {
 norm_scale_trace<<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)x,(GM_ADDR)y);return 0;
}
