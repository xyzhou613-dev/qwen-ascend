#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
template<int TILE> __global__ __aicore__ void state_post(GM_ADDR current,GM_ADDR decay,GM_ADDR delta,GM_ADDR output) {
 GlobalTensor<float> gs,gd,ga,go;gs.SetGlobalBuffer((__gm__ float*)current);gd.SetGlobalBuffer((__gm__ float*)decay);ga.SetGlobalBuffer((__gm__ float*)delta);go.SetGlobalBuffer((__gm__ float*)output);
 TPipe pipe;TBuf<TPosition::VECCALC> sb,ab;pipe.InitBuffer(sb,TILE*4);pipe.InitBuffer(ab,TILE*4);auto s=sb.Get<float>();auto a=ab.Get<float>();
 for(int tile=GetBlockIdx();tile<48*16384/TILE;tile+=GetBlockNum()) {
  int offset=tile*TILE;float scale=gd.GetValue(offset/16384);
  DataCopy(s,gs[offset],TILE);DataCopy(a,ga[offset],TILE);SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  // Explicitly rounded FP32 multiply, then add. Do not use fused multiply-add.
  Muls(s,s,scale,TILE);PipeBarrier<PIPE_V>();Add(s,s,a,TILE);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(go[offset],s,TILE);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
template<int TILE> __global__ __aicore__ void output_post(GM_ADDR left,GM_ADDR right,GM_ADDR output) {
 GlobalTensor<float> gl,gr;GlobalTensor<bfloat16_t> go;gl.SetGlobalBuffer((__gm__ float*)left);gr.SetGlobalBuffer((__gm__ float*)right);go.SetGlobalBuffer((__gm__ bfloat16_t*)output);
 TPipe pipe;TBuf<TPosition::VECCALC> lb,rb,ob;pipe.InitBuffer(lb,TILE*4);pipe.InitBuffer(rb,TILE*4);pipe.InitBuffer(ob,TILE*2);auto l=lb.Get<float>();auto r=rb.Get<float>();auto o=ob.Get<bfloat16_t>();
 for(int tile=GetBlockIdx();tile<48*64*128/TILE;tile+=GetBlockNum()) {
  int offset=tile*TILE;DataCopy(l,gl[offset],TILE);DataCopy(r,gr[offset],TILE);SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  Add(l,l,r,TILE);PipeBarrier<PIPE_V>();Cast(o,l,RoundMode::CAST_RINT,TILE);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(go[offset],o,TILE);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" int fnx_prefill_state(void*s,void*d,void*a,void*o,int tile,void*stream) {
 if(tile==2048)state_post<2048><<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)s,(GM_ADDR)d,(GM_ADDR)a,(GM_ADDR)o);
 else if(tile==4096)state_post<4096><<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)s,(GM_ADDR)d,(GM_ADDR)a,(GM_ADDR)o);
 else return -1;return 0;
}
extern "C" int fnx_prefill_output(void*l,void*r,void*o,int tile,void*stream) {
 if(tile==2048)output_post<2048><<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)l,(GM_ADDR)r,(GM_ADDR)o);
 else if(tile==4096)output_post<4096><<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)l,(GM_ADDR)r,(GM_ADDR)o);
 else return -1;return 0;
}
