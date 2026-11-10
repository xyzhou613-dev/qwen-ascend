#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
namespace {constexpr int Tile=4096;}
extern "C" __global__ __aicore__ void conv_prefill(GM_ADDR sequence,GM_ADDR weights,GM_ADDR output,int dimension,int tokens) {
 GlobalTensor<bfloat16_t> gx,gw,gy;gx.SetGlobalBuffer((__gm__ bfloat16_t*)sequence);gw.SetGlobalBuffer((__gm__ bfloat16_t*)weights);gy.SetGlobalBuffer((__gm__ bfloat16_t*)output);
 TPipe pipe;TBuf<TPosition::VECCALC> xb,fb,rb,wb,wfb;
 pipe.InitBuffer(xb,4*Tile*2);pipe.InitBuffer(fb,4*Tile*4);pipe.InitBuffer(rb,Tile*2);pipe.InitBuffer(wb,32);pipe.InitBuffer(wfb,64);
 auto x=xb.Get<bfloat16_t>();auto f=fb.Get<float>();auto rounded=rb.Get<bfloat16_t>();auto w=wb.Get<bfloat16_t>();auto wf=wfb.Get<float>();
 const int columns=(tokens+Tile-1)/Tile;
 for(int task=GetBlockIdx();task<dimension*columns;task+=GetBlockNum()) {
  const int channel=task/columns,begin=(task%columns)*Tile,count=tokens-begin<Tile?tokens-begin:Tile;
  DataCopy(w,gw[(channel/4)*16],16);
  DataCopyExtParams cp{1,static_cast<uint32_t>(count*2),0,0,0};DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
  for(int j=0;j<4;++j)DataCopyPad(x[j*Tile],gx[static_cast<int64_t>(channel)*(tokens+3)+begin+j],cp,pad);
  SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  Cast(wf,w,RoundMode::CAST_NONE,16);
  for(int j=0;j<4;++j)Cast(f[j*Tile],x[j*Tile],RoundMode::CAST_NONE,count);
  SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
  for(int j=0;j<4;++j) {
   const float scale=wf.GetValue((channel%4)*4+j);
   Muls(f[j*Tile],f[j*Tile],scale,count);PipeBarrier<PIPE_V>();
   Cast(rounded,f[j*Tile],RoundMode::CAST_RINT,count);PipeBarrier<PIPE_V>();
   Cast(f[j*Tile],rounded,RoundMode::CAST_NONE,count);PipeBarrier<PIPE_V>();
  }
  Add(f,f,f[Tile],count);Add(f[2*Tile],f[2*Tile],f[3*Tile],count);PipeBarrier<PIPE_V>();
  Add(f,f,f[2*Tile],count);PipeBarrier<PIPE_V>();Cast(rounded,f,RoundMode::CAST_RINT,count);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
  DataCopyPad(gy[static_cast<int64_t>(channel)*tokens+begin],rounded,cp);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);
  SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" int fnx_conv_prefill(void* seq,void* weights,void* out,int dimension,int tokens,void* stream) {
 if(dimension<4||dimension%4||tokens<1)return -1;
 conv_prefill<<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)seq,(GM_ADDR)weights,(GM_ADDR)out,dimension,tokens);return 0;
}
