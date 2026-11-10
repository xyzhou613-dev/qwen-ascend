#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void expert_gather_layout(
 GM_ADDR weights,GM_ADDR scales,GM_ADDR ids,GM_ADDR wi,GM_ADDR si,GM_ADDR ow,GM_ADDR os,
 int K,int G,int N,int rows_w,int overlap) {
 GlobalTensor<float> gw,gs,yw,ys;GlobalTensor<uint32_t> gwi,gsi;GlobalTensor<int64_t> gi;
 gw.SetGlobalBuffer((__gm__ float*)weights);gs.SetGlobalBuffer((__gm__ float*)scales);
 gwi.SetGlobalBuffer((__gm__ uint32_t*)wi);gsi.SetGlobalBuffer((__gm__ uint32_t*)si);
 yw.SetGlobalBuffer((__gm__ float*)ow);ys.SetGlobalBuffer((__gm__ float*)os);gi.SetGlobalBuffer((__gm__ int64_t*)ids);
 int experts[10];for(int e=0;e<10;++e)experts[e]=static_cast<int>(gi.GetValue(e));
 const int max_items=rows_w*10*(N/8);
 TPipe pipe;TBuf<TPosition::VECCALC> xb,yb,ib;
 pipe.InitBuffer(xb,max_items*4);pipe.InitBuffer(yb,max_items*4);pipe.InitBuffer(ib,max_items*4);
 auto x=xb.Get<float>();auto y=yb.Get<float>();auto index=ib.Get<uint32_t>();
 for(int part=0;part<2;++part) {
  const int rows=part?G:K,columns=part?N/2:N/8,rt=part?rows_w/4:rows_w;
  const int items=rt*10*columns;bool pending=false;
  if(part)DataCopy(index,gsi,items);else DataCopy(index,gwi,items);
  for(int t=GetBlockIdx();t<rows/rt;t+=GetBlockNum()) {
   const int begin=t*rt;
   for(int e=0;e<10;++e) {
    const int64_t offset=(static_cast<int64_t>(experts[e])*rows+begin)*columns;
    if(part)DataCopy(x[e*rt*columns],gs[offset],rt*columns);
    else DataCopy(x[e*rt*columns],gw[offset],rt*columns);
   }
   SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
   if(pending){WaitFlag<HardEvent::MTE3_V>(0);pending=false;}
   Gather(y,x,index,0,items);
   SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
   if(part)DataCopy(ys[static_cast<int64_t>(begin)*10*columns],y,items);
   else DataCopy(yw[static_cast<int64_t>(begin)*10*columns],y,items);
   if(overlap){SetFlag<HardEvent::MTE3_V>(0);pending=true;}
   else {SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);}
   SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
  }
  if(pending)WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" int fnx_expert_gather(void*w,void*s,void*i,void*wi,void*si,void*ow,void*os,int K,int G,int N,int rows,int overlap,void*stream) {
 if(K!=2560||G!=40||N!=1280||(rows!=4&&rows!=8)||overlap<0||overlap>1)return -1;
 expert_gather_layout<<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)i,(GM_ADDR)wi,(GM_ADDR)si,(GM_ADDR)ow,(GM_ADDR)os,K,G,N,rows,overlap);return 0;
}
