#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void down_gather(GM_ADDR weights,GM_ADDR scales,GM_ADDR ids,GM_ADDR ow,GM_ADDR os) {
 GlobalTensor<uint8_t> gw,gs,yw,ys;GlobalTensor<int64_t> gi;
 gw.SetGlobalBuffer((__gm__ uint8_t*)weights);gs.SetGlobalBuffer((__gm__ uint8_t*)scales);yw.SetGlobalBuffer((__gm__ uint8_t*)ow);ys.SetGlobalBuffer((__gm__ uint8_t*)os);gi.SetGlobalBuffer((__gm__ int64_t*)ids);
 const int slot=GetBlockIdx()%10,worker=GetBlockIdx()/10,expert=static_cast<int>(gi.GetValue(slot));
 TPipe pipe;TBuf<TPosition::VECCALC> buffer;pipe.InitBuffer(buffer,65536);auto tmp=buffer.Get<uint8_t>();
 for(int part=0;part<2;++part) {
  const int stride=part?51200:819200,chunk=stride/4,begin=worker*chunk;
  for(int offset=0;offset<chunk;offset+=65536) {
   const int count=chunk-offset<65536?chunk-offset:65536;
   if(part)DataCopy(tmp,gs[static_cast<int64_t>(expert)*stride+begin+offset],count);else DataCopy(tmp,gw[static_cast<int64_t>(expert)*stride+begin+offset],count);
   SetFlag<HardEvent::MTE2_MTE3>(0);WaitFlag<HardEvent::MTE2_MTE3>(0);
   if(part)DataCopy(ys[slot*stride+begin+offset],tmp,count);else DataCopy(yw[slot*stride+begin+offset],tmp,count);
   SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);
  }
 }
}
extern "C" int fnx_down_gather(void*w,void*s,void*i,void*ow,void*os,void*stream) {
 down_gather<<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)i,(GM_ADDR)ow,(GM_ADDR)os);return 0;
}
