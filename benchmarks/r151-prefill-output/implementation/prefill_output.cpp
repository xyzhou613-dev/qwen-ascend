#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
template<int TILE> __global__ __aicore__ void output_scatter(GM_ADDR left,GM_ADDR right,GM_ADDR output) {
 GlobalTensor<float> gl,gr;GlobalTensor<bfloat16_t> go;gl.SetGlobalBuffer((__gm__ float*)left);gr.SetGlobalBuffer((__gm__ float*)right);go.SetGlobalBuffer((__gm__ bfloat16_t*)output);
 TPipe pipe;TBuf<TPosition::VECCALC> lb,rb,ob;pipe.InitBuffer(lb,TILE*4);pipe.InitBuffer(rb,TILE*4);pipe.InitBuffer(ob,TILE*2);auto l=lb.Get<float>();auto r=rb.Get<float>();auto o=ob.Get<bfloat16_t>();
 DataCopyParams store{TILE/128,8,0,47*8};
 for(int tile=GetBlockIdx();tile<48*64*128/TILE;tile+=GetBlockNum()) {
  int offset=tile*TILE,head=offset/8192,row=(offset%8192)/128;
  DataCopy(l,gl[offset],TILE);DataCopy(r,gr[offset],TILE);SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  Add(l,l,r,TILE);PipeBarrier<PIPE_V>();Cast(o,l,RoundMode::CAST_RINT,TILE);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(go[(row*48+head)*128],o,store);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
template<int HEADS> __global__ __aicore__ void output_gather(GM_ADDR left,GM_ADDR right,GM_ADDR output) {
 constexpr int TILE=HEADS*128,GROUPS=48/HEADS;
 GlobalTensor<float> gl,gr;GlobalTensor<bfloat16_t> go;gl.SetGlobalBuffer((__gm__ float*)left);gr.SetGlobalBuffer((__gm__ float*)right);go.SetGlobalBuffer((__gm__ bfloat16_t*)output);
 TPipe pipe;TBuf<TPosition::VECCALC> lb,rb,ob;pipe.InitBuffer(lb,TILE*4);pipe.InitBuffer(rb,TILE*4);pipe.InitBuffer(ob,TILE*2);auto l=lb.Get<float>();auto r=rb.Get<float>();auto o=ob.Get<bfloat16_t>();
 DataCopyParams load{HEADS,16,63*16,0};
 for(int task=GetBlockIdx();task<64*GROUPS;task+=GetBlockNum()) {
  int row=task/GROUPS,head=(task%GROUPS)*HEADS,offset=head*8192+row*128;
  DataCopy(l,gl[offset],load);DataCopy(r,gr[offset],load);SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  Add(l,l,r,TILE);PipeBarrier<PIPE_V>();Cast(o,l,RoundMode::CAST_RINT,TILE);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(go[(row*48+head)*128],o,TILE);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" int fnx_prefill_output_into(void*l,void*r,void*o,int mode,void*stream) {
 if(mode==1)output_scatter<2048><<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)l,(GM_ADDR)r,(GM_ADDR)o);
 else if(mode==2)output_scatter<4096><<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)l,(GM_ADDR)r,(GM_ADDR)o);
 else if(mode==3)output_gather<16><<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)l,(GM_ADDR)r,(GM_ADDR)o);
 else if(mode==4)output_gather<8><<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)l,(GM_ADDR)r,(GM_ADDR)o);
 else if(mode==5)output_gather<48><<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)l,(GM_ADDR)r,(GM_ADDR)o);
 else return -1;return 0;
}
