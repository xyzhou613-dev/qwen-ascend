#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
namespace { constexpr int Tile=2048; constexpr float Negative=-1.0e30f; }
extern "C" __global__ __aicore__ void index_post(
 GM_ADDR source,GM_ADDR positions,GM_ADDR destination,int tokens,int groups,int heads,int mode) {
 GlobalTensor<float> gx,gy;GlobalTensor<int64_t> gp;
 gx.SetGlobalBuffer((__gm__ float*)source);gy.SetGlobalBuffer((__gm__ float*)destination);gp.SetGlobalBuffer((__gm__ int64_t*)positions);
 TPipe pipe;TBuf<TPosition::VECCALC> scratch;pipe.InitBuffer(scratch,4*Tile*4);
 auto a=scratch.Get<float>();auto b=a[Tile];auto c=a[2*Tile];auto d=a[3*Tile];
 const int columns=groups/Tile;
 for(int task=GetBlockIdx();task<tokens*columns;task+=GetBlockNum()) {
  const int row=task/columns,col=(task%columns)*Tile;
  const int64_t remaining=(gp.GetValue(row)+1)/4-col;
  const int valid=remaining>Tile?Tile:(remaining>0?remaining:0),load=(valid+7)/8*8;
  if(load) {
   const int64_t offset=static_cast<int64_t>(row)*heads*groups+col;
   DataCopy(a,gx[offset],load);
   if(heads==4){DataCopy(b,gx[offset+groups],load);DataCopy(c,gx[offset+2*groups],load);DataCopy(d,gx[offset+3*groups],load);}
   SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
   if(heads==4) {
    Relu(a,a,load);Relu(b,b,load);Relu(c,c,load);Relu(d,d,load);PipeBarrier<PIPE_V>();
    Add(a,a,b,load);PipeBarrier<PIPE_V>();Add(a,a,c,load);PipeBarrier<PIPE_V>();Add(a,a,d,load);PipeBarrier<PIPE_V>();
   }
   if(mode==0)Muls(a,a,0.08838834764831844055f,load);
   else {Duplicate(b,11.31370849898476039f,load);PipeBarrier<PIPE_V>();Div(a,a,b,load);}
   PipeBarrier<PIPE_V>();
  }
  if(load<Tile)Duplicate(a[load],Negative,Tile-load);
  if(load!=valid) {
   SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
   for(int j=valid;j<load;++j)a.SetValue(j,Negative);
   SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);
  }
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(gy[static_cast<int64_t>(row)*groups+col],a,Tile);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" int fnx_index_post(void*x,void*p,void*y,int tokens,int groups,int heads,int mode,void*stream) {
 if(tokens<1||groups<2048||groups%2048||(heads!=1&&heads!=4)||mode<0||mode>1)return -1;
 const int tasks=tokens*(groups/Tile),blocks=tasks<40?tasks:40;
 index_post<<<blocks,nullptr,(aclrtStream)stream>>>((GM_ADDR)x,(GM_ADDR)p,(GM_ADDR)y,tokens,groups,heads,mode);return 0;
}
