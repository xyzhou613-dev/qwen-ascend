#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
template<int TILE,bool LOCAL> __global__ __aicore__ void state_sync(GM_ADDR current,GM_ADDR decay,GM_ADDR delta,GM_ADDR output) {
 GlobalTensor<float> gs,gd,ga,go;gs.SetGlobalBuffer((__gm__ float*)current);gd.SetGlobalBuffer((__gm__ float*)decay);ga.SetGlobalBuffer((__gm__ float*)delta);go.SetGlobalBuffer((__gm__ float*)output);
 TPipe pipe;TBuf<TPosition::VECCALC> sb,ab,db;pipe.InitBuffer(sb,TILE*4);pipe.InitBuffer(ab,TILE*4);pipe.InitBuffer(db,192);
 auto s=sb.Get<float>();auto a=ab.Get<float>();auto d=db.Get<float>();
 if constexpr(LOCAL){DataCopy(d,gd,48);SetFlag<HardEvent::MTE2_S>(0);WaitFlag<HardEvent::MTE2_S>(0);}
 for(int tile=GetBlockIdx();tile<48*16384/TILE;tile+=GetBlockNum()) {
  int offset=tile*TILE;float scale=LOCAL?d.GetValue(offset/16384):gd.GetValue(offset/16384);
  DataCopy(s,gs[offset],TILE);DataCopy(a,ga[offset],TILE);SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  Muls(s,s,scale,TILE);PipeBarrier<PIPE_V>();Add(s,s,a,TILE);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(go[offset],s,TILE);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
template<int TILE> __global__ __aicore__ void state_pipelined(GM_ADDR current,GM_ADDR decay,GM_ADDR delta,GM_ADDR output) {
 GlobalTensor<float> gs,gd,ga,go;gs.SetGlobalBuffer((__gm__ float*)current);gd.SetGlobalBuffer((__gm__ float*)decay);ga.SetGlobalBuffer((__gm__ float*)delta);go.SetGlobalBuffer((__gm__ float*)output);
 TPipe pipe;TQue<TPosition::VECIN,2> input;TQue<TPosition::VECOUT,2> result;TBuf<TPosition::VECCALC> db;
 pipe.InitBuffer(input,2,TILE*8);pipe.InitBuffer(result,2,TILE*4);pipe.InitBuffer(db,192);auto d=db.Get<float>();
 DataCopy(d,gd,48);SetFlag<HardEvent::MTE2_S>(0);WaitFlag<HardEvent::MTE2_S>(0);
 for(int tile=GetBlockIdx();tile<48*16384/TILE;tile+=GetBlockNum()) {
  int offset=tile*TILE;float scale=d.GetValue(offset/16384);
  auto xy=input.AllocTensor<float>();DataCopy(xy,gs[offset],TILE);DataCopy(xy[TILE],ga[offset],TILE);input.EnQue(xy);
  auto values=input.DeQue<float>();auto out=result.AllocTensor<float>();
  // Two separate FP32 operations; preserve the multiply's rounding.
  Muls(out,values,scale,TILE);PipeBarrier<PIPE_V>();Add(out,out,values[TILE],TILE);
  result.EnQue(out);input.FreeTensor(values);
  auto ready=result.DeQue<float>();DataCopy(go[offset],ready,TILE);result.FreeTensor(ready);
 }
}
extern "C" int fnx_state_update(void*s,void*d,void*a,void*o,int mode,void*stream) {
 if(mode==0)state_sync<4096,false><<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)s,(GM_ADDR)d,(GM_ADDR)a,(GM_ADDR)o);
 else if(mode==1)state_sync<2048,true><<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)s,(GM_ADDR)d,(GM_ADDR)a,(GM_ADDR)o);
 else if(mode==2)state_sync<4096,true><<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)s,(GM_ADDR)d,(GM_ADDR)a,(GM_ADDR)o);
 else if(mode==3)state_pipelined<2048><<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)s,(GM_ADDR)d,(GM_ADDR)a,(GM_ADDR)o);
 else if(mode==4)state_pipelined<4096><<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)s,(GM_ADDR)d,(GM_ADDR)a,(GM_ADDR)o);
 else if(mode==5)state_sync<8192,true><<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)s,(GM_ADDR)d,(GM_ADDR)a,(GM_ADDR)o);
 else if(mode==6)state_pipelined<1024><<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)s,(GM_ADDR)d,(GM_ADDR)a,(GM_ADDR)o);
 else return -1;return 0;
}
