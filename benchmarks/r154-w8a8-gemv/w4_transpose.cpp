#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;

// Exact signed-INT4 matrix transpose, tiled in 128x128 elements.
// A full-size scratch destination is required; source is never overwritten.
extern "C" __global__ __aicore__ void packed_transpose(
 GM_ADDR input,GM_ADDR output,int experts,int rows,int cols) {
 constexpr int T=128;
 GlobalTensor<int8_t> gx,gy;gx.SetGlobalBuffer((__gm__ int8_t*)input);gy.SetGlobalBuffer((__gm__ int8_t*)output);
 TPipe pipe;TBuf<TPosition::VECCALC> ib,hb,tb,ob;
 pipe.InitBuffer(ib,T*T/2);pipe.InitBuffer(hb,T*T*2);pipe.InitBuffer(tb,T*T*2);pipe.InitBuffer(ob,T*T/2);
 auto in=ib.Get<int8_t>();auto h=hb.Get<half>();auto trans=tb.Get<half>();auto out=ob.Get<int8_t>();
 const int rt=rows/T,ct=cols/T;
 for(int tile=GetBlockIdx();tile<experts*rt*ct;tile+=GetBlockNum()) {
  int e=tile/(rt*ct),rem=tile%(rt*ct),r=(rem/ct)*T,c=(rem%ct)*T;
  DataCopyParams load{T,T/2/32,static_cast<uint16_t>(cols/2/32-T/2/32),0};
  DataCopy(in,gx[(static_cast<int64_t>(e)*rows+r)*(cols/2)+c/2],load);
  SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  Cast(h,in.ReinterpretCast<int4b_t>(),RoundMode::CAST_NONE,T*T);PipeBarrier<PIPE_V>();
  for(int cb=0;cb<T;cb+=16) {
   uint64_t src[16],dst[16];
   for(int i=0;i<16;++i) {
    src[i]=reinterpret_cast<uint64_t>(h[i*T+cb].GetPhyAddr());
    dst[i]=reinterpret_cast<uint64_t>(trans[(cb+i)*T].GetPhyAddr());
   }
   TransDataTo5HDParams params{false,false,8,1,128};
   TransDataTo5HD<half>(dst,src,params);
  }
  PipeBarrier<PIPE_V>();Cast(out.ReinterpretCast<int4b_t>(),trans,RoundMode::CAST_RINT,T*T);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
  DataCopyParams store{T,T/2/32,0,static_cast<uint16_t>(rows/2/32-T/2/32)};
  DataCopy(gy[(static_cast<int64_t>(e)*cols+c)*(rows/2)+r/2],out,store);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);
  SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" int fnx_w4_transpose(void*x,void*y,int e,int rows,int cols,void*stream) {
 if(e<1||e>512||rows<128||cols<128||rows%128||cols%128||x==y)return -1;
 packed_transpose<<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)x,(GM_ADDR)y,e,rows,cols);return 0;
}
