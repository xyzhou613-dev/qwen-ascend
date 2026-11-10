#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void raw_gemv(
 GM_ADDR input,GM_ADDR weight,GM_ADDR output,int N,int K,int row_batch) {
 GlobalTensor<bfloat16_t> gx,gw,gy;
 gx.SetGlobalBuffer((__gm__ bfloat16_t*)input);gw.SetGlobalBuffer((__gm__ bfloat16_t*)weight);gy.SetGlobalBuffer((__gm__ bfloat16_t*)output);
 TPipe pipe;TBuf<TPosition::VECCALC> xb,xf,wb,wf,rb,yb,tmp;
 pipe.InitBuffer(xb,K*2);pipe.InitBuffer(xf,K*4);pipe.InitBuffer(wb,K*row_batch*2);
 pipe.InitBuffer(wf,K*4);pipe.InitBuffer(rb,64);pipe.InitBuffer(yb,32);pipe.InitBuffer(tmp,4096);
 auto x=xb.Get<bfloat16_t>();auto x32=xf.Get<float>();auto w=wb.Get<bfloat16_t>();auto w32=wf.Get<float>();
 auto result=rb.Get<float>();auto y=yb.Get<bfloat16_t>();auto scratch=tmp.Get<float>();
 DataCopy(x,gx,K);SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
 Cast(x32,x,RoundMode::CAST_NONE,K);PipeBarrier<PIPE_V>();
 for(int tile=GetBlockIdx();tile<(N+15)/16;tile+=GetBlockNum()) {
  const int begin=tile*16;Duplicate(result,0.0f,16);PipeBarrier<PIPE_V>();
  for(int r=0;r<16;r+=row_batch) {
   int count=N-begin-r;if(count>row_batch)count=row_batch;if(count<=0)break;
   DataCopy(w,gw[static_cast<int64_t>(begin+r)*K],count*K);
   SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
   for(int j=0;j<count;++j) {
    Cast(w32,w[j*K],RoundMode::CAST_NONE,K);PipeBarrier<PIPE_V>();
    Mul(w32,w32,x32,K);PipeBarrier<PIPE_V>();
    ReduceSum(result[r+j],w32,scratch,K);PipeBarrier<PIPE_V>();
   }
   SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
  }
  Cast(y,result,RoundMode::CAST_RINT,16);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
  DataCopy(gy[begin],y,16);
  SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" int fnx_raw_gemv(void*x,void*w,void*y,int N,int K,int blocks,int batch,void*stream) {
 if(N<1||K<32||K%32||K>10240||batch<1||batch>8||K*(10+batch*2)+8192>196608)return -1;
 int tasks=(N+15)/16;if(blocks>tasks)blocks=tasks;
 raw_gemv<<<blocks,nullptr,(aclrtStream)stream>>>((GM_ADDR)x,(GM_ADDR)w,(GM_ADDR)y,N,K,batch);return 0;
}
