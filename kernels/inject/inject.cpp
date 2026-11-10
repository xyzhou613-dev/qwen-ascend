#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void residual_inject(GM_ADDR h,GM_ADDR y,GM_ADDR gate,GM_ADDR out,int batch) {
 GlobalTensor<bfloat16_t> gh,gy,go;GlobalTensor<uint16_t> gg;
 gh.SetGlobalBuffer((__gm__ bfloat16_t*)h);gy.SetGlobalBuffer((__gm__ bfloat16_t*)y);gg.SetGlobalBuffer((__gm__ uint16_t*)gate);go.SetGlobalBuffer((__gm__ bfloat16_t*)out);
 TPipe pipe;TBuf<TPosition::VECCALC> hb,yb,pb,ob,hf,yf;
 pipe.InitBuffer(hb,512);pipe.InitBuffer(yb,512);pipe.InitBuffer(pb,512);pipe.InitBuffer(ob,512);pipe.InitBuffer(hf,1024);pipe.InitBuffer(yf,1024);
 auto hv=hb.Get<bfloat16_t>();auto yv=yb.Get<bfloat16_t>();auto rounded=pb.Get<bfloat16_t>();auto ov=ob.Get<bfloat16_t>();auto h32=hf.Get<float>();auto p32=yf.Get<float>();
 for(int task=GetBlockIdx();task<batch*40;task+=GetBlockNum()) {
  const int token=task/40,block=task%40,branch=block/10,column=(block%10)*256;
  const int64_t offset=static_cast<int64_t>(token)*10240+block*256;
  DataCopy(hv,gh[offset],256);DataCopy(yv,gy[static_cast<int64_t>(token)*2560+column],256);
  union {uint32_t bits;float value;} scale_bits;
  scale_bits.bits=static_cast<uint32_t>(gg.GetValue(token*4+branch))<<16;
  float scale=scale_bits.value;
  SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  Cast(h32,hv,RoundMode::CAST_NONE,256);Cast(p32,yv,RoundMode::CAST_NONE,256);PipeBarrier<PIPE_V>();
  Muls(p32,p32,scale,256);PipeBarrier<PIPE_V>();
  Cast(rounded,p32,RoundMode::CAST_RINT,256);PipeBarrier<PIPE_V>();
  Cast(p32,rounded,RoundMode::CAST_NONE,256);PipeBarrier<PIPE_V>();
  Add(p32,h32,p32,256);PipeBarrier<PIPE_V>();Cast(ov,p32,RoundMode::CAST_RINT,256);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(go[offset],ov,256);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);
  SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" int fnx_inject(void*h,void*y,void*g,void*out,int batch,void*stream) {
 if(batch<1)return -1;
 residual_inject<<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)h,(GM_ADDR)y,(GM_ADDR)g,(GM_ADDR)out,batch);return 0;
}
