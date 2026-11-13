#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;

// Production expert-major INT4 [E,640,2560/8] and cached BF16 scales
// [E,10,2560]. No weight repack or selected-weight GM temporary.
template<int TILE> __aicore__ inline void run_down(
 GM_ADDR weights,GM_ADDR scales,GM_ADDR ids,GM_ADDR input,GM_ADDR output) {
 constexpr int K=640,N=2560,G=10;
 const int slot=GetBlockIdx()%10,worker=GetBlockIdx()/10;
 GlobalTensor<int8_t> gw;GlobalTensor<bfloat16_t> gs,gx,gy;GlobalTensor<int64_t> gi;
 gw.SetGlobalBuffer((__gm__ int8_t*)weights);gs.SetGlobalBuffer((__gm__ bfloat16_t*)scales);
 gx.SetGlobalBuffer((__gm__ bfloat16_t*)input);gi.SetGlobalBuffer((__gm__ int64_t*)ids);
 gy.SetGlobalBuffer((__gm__ bfloat16_t*)output);
 const int expert=static_cast<int>(gi.GetValue(slot));
 TPipe pipe;TBuf<TPosition::VECCALC> qb,hb,fb,xb,xfb,brb,sb,sfb,ab,ob;
 pipe.InitBuffer(qb,64*TILE/2);pipe.InitBuffer(hb,64*TILE*2);pipe.InitBuffer(fb,64*TILE*4);
 pipe.InitBuffer(xb,K*2);pipe.InitBuffer(xfb,K*4);pipe.InitBuffer(brb,K*8*4);
 pipe.InitBuffer(sb,TILE*2);pipe.InitBuffer(sfb,TILE*4);pipe.InitBuffer(ab,TILE*4);pipe.InitBuffer(ob,TILE*2);
 auto q=qb.Get<int8_t>();auto h=hb.Get<half>();auto f=fb.Get<float>();auto x=xb.Get<bfloat16_t>();
 auto xf=xfb.Get<float>();auto br=brb.Get<float>();auto s=sb.Get<bfloat16_t>();
 auto sf=sfb.Get<float>();auto accum=ab.Get<float>();auto out=ob.Get<bfloat16_t>();
 DataCopy(x,gx[slot*K],K);SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
 Cast(xf,x,RoundMode::CAST_NONE,K);PipeBarrier<PIPE_V>();
 Brcb(br,xf,K/8,{1,8});PipeBarrier<PIPE_V>();
 for(int row=worker*TILE;row<N;row+=4*TILE) {
  Duplicate(accum,0.0f,TILE);PipeBarrier<PIPE_V>();
  for(int group=0;group<G;++group) {
   DataCopyParams load{64,TILE/64,(N-TILE)/64,0};
   DataCopy(q,gw[(static_cast<int64_t>(expert)*K+group*64)*(N/2)+row/2],load);
   DataCopy(s,gs[(static_cast<int64_t>(expert)*G+group)*N+row],TILE);
   SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
   Cast(h,q.ReinterpretCast<int4b_t>(),RoundMode::CAST_NONE,64*TILE);PipeBarrier<PIPE_V>();
   Cast(f,h,RoundMode::CAST_NONE,64*TILE);Cast(sf,s,RoundMode::CAST_NONE,TILE);PipeBarrier<PIPE_V>();
   // Each scalar activation is broadcast into one 32-byte block by Brcb.
   // src1 block stride 0 broadcasts that block across all columns of a row.
   for(int col=0;col<TILE;col+=64)
    Mul(f[col],f[col],br[group*64*8],64,64,{1,1,0,TILE/8,TILE/8,1});
   PipeBarrier<PIPE_V>();
   for(int rows=32;rows>0;rows/=2) {
    Add(f,f,f[rows*TILE],rows*TILE);PipeBarrier<PIPE_V>();
   }
   Mul(f,f,sf,TILE);PipeBarrier<PIPE_V>();
   Add(accum,accum,f,TILE);PipeBarrier<PIPE_V>();
   SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
  }
  Cast(out,accum,RoundMode::CAST_RINT,TILE);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
  DataCopy(gy[slot*N+row],out,TILE);
  SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" __global__ __aicore__ void down64(GM_ADDR w,GM_ADDR s,GM_ADDR i,GM_ADDR x,GM_ADDR y) {run_down<64>(w,s,i,x,y);}
extern "C" __global__ __aicore__ void down128(GM_ADDR w,GM_ADDR s,GM_ADDR i,GM_ADDR x,GM_ADDR y) {run_down<128>(w,s,i,x,y);}
extern "C" __global__ __aicore__ void down256(GM_ADDR w,GM_ADDR s,GM_ADDR i,GM_ADDR x,GM_ADDR y) {run_down<256>(w,s,i,x,y);}
extern "C" int fnx_w4_kmajor(void*w,void*s,void*i,void*x,void*y,int tile,void*stream) {
 if(tile==64)down64<<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)i,(GM_ADDR)x,(GM_ADDR)y);
 else if(tile==128)down128<<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)i,(GM_ADDR)x,(GM_ADDR)y);
 else if(tile==256)down256<<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)i,(GM_ADDR)x,(GM_ADDR)y);
 else return -1;
 return 0;
}
#include "diag_epilogue.cpp"
#include "fused_down.cpp"
