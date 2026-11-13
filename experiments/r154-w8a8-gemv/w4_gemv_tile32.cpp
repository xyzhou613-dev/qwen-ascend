#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;

// Row-major packed INT4: [E,N,K/2], scale BF16 [E,N,round_up(K/64,8)].
// No selected-weight/scales GM temporary. Each of 40 cores owns one of four
// output-channel partitions for one of ten dynamically selected experts.
extern "C" __global__ __aicore__ void w4_gemv_direct(
 GM_ADDR weights,GM_ADDR scales,GM_ADDR ids,GM_ADDR input,GM_ADDR output,int K,int N,int per_expert) {
 const int TILE=K==640?32:8;
 const int G=K/64,GP=(G+7)/8*8;
 const int slot=GetBlockIdx()%10,worker=GetBlockIdx()/10;
 GlobalTensor<int8_t> gw;GlobalTensor<bfloat16_t> gs,gx;GlobalTensor<int64_t> gi;GlobalTensor<float> gy;
 gw.SetGlobalBuffer((__gm__ int8_t*)weights);gs.SetGlobalBuffer((__gm__ bfloat16_t*)scales);
 gx.SetGlobalBuffer((__gm__ bfloat16_t*)input);gi.SetGlobalBuffer((__gm__ int64_t*)ids);gy.SetGlobalBuffer((__gm__ float*)output);
 const int expert=static_cast<int>(gi.GetValue(slot));
 TPipe pipe;TBuf<TPosition::VECCALC> qb,hb,fb,xb,xfbuf,sb,sfbuf,pb,ob;
 pipe.InitBuffer(qb,TILE*K/2);pipe.InitBuffer(hb,TILE*K*2);pipe.InitBuffer(fb,TILE*K*4);
 pipe.InitBuffer(xb,K*2);pipe.InitBuffer(xfbuf,K*4);
 pipe.InitBuffer(sb,TILE*GP*2);pipe.InitBuffer(sfbuf,TILE*GP*4);pipe.InitBuffer(pb,TILE*GP*4);pipe.InitBuffer(ob,TILE*4);
 auto qp=qb.Get<int8_t>();auto h=hb.Get<half>();auto f=fb.Get<float>();auto x=xb.Get<bfloat16_t>();auto xf=xfbuf.Get<float>();
 auto s=sb.Get<bfloat16_t>();auto sf=sfbuf.Get<float>();auto partial=pb.Get<float>();auto out=ob.Get<float>();
 DataCopy(x,gx[static_cast<int64_t>(per_expert?slot:0)*K],K);
 SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
 Cast(xf,x,RoundMode::CAST_NONE,K);PipeBarrier<PIPE_V>();
 Duplicate(partial,0.0f,TILE*GP);PipeBarrier<PIPE_V>();
 for(int row=worker*TILE;row<N;row+=4*TILE) {
  DataCopy(qp,gw[(static_cast<int64_t>(expert)*N+row)*(K/2)],TILE*K/2);
  DataCopy(s,gs[(static_cast<int64_t>(expert)*N+row)*GP],TILE*GP);
  SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  Cast(h,qp.ReinterpretCast<int4b_t>(),RoundMode::CAST_NONE,TILE*K);PipeBarrier<PIPE_V>();
  Cast(f,h,RoundMode::CAST_NONE,TILE*K);Cast(sf,s,RoundMode::CAST_NONE,TILE*GP);PipeBarrier<PIPE_V>();
  for(int r=0;r<TILE;++r)Mul(f[r*K],f[r*K],xf,K);
  PipeBarrier<PIPE_V>();
  for(int r=0;r<TILE;++r)WholeReduceSum(partial[r*GP],f[r*K],64,G,1,1,8);
  PipeBarrier<PIPE_V>();Mul(partial,partial,sf,TILE*GP);PipeBarrier<PIPE_V>();
  WholeReduceSum(out,partial,G,TILE,1,1,GP/8);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
  DataCopy(gy[static_cast<int64_t>(slot)*N+row],out,TILE);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);
  SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}

extern "C" int fnx_w4_gemv(void*w,void*s,void*i,void*x,void*y,int K,int N,int per_expert,void*stream) {
 if((K!=2560||N!=1280)&&(K!=640||N!=2560))return -1;
 if(per_expert!=0&&per_expert!=1)return -2;
 w4_gemv_direct<<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)i,(GM_ADDR)x,(GM_ADDR)y,K,N,per_expert);
 return 0;
}
