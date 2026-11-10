#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void gather_pair(GM_ADDR weights,GM_ADDR scales,GM_ADDR ids,GM_ADDR ow,GM_ADDR os,int K,int G,int N) {
 GlobalTensor<uint8_t> gw,gs,yw,ys;GlobalTensor<int64_t> gi;
 gw.SetGlobalBuffer((__gm__ uint8_t*)weights);gs.SetGlobalBuffer((__gm__ uint8_t*)scales);
 yw.SetGlobalBuffer((__gm__ uint8_t*)ow);ys.SetGlobalBuffer((__gm__ uint8_t*)os);gi.SetGlobalBuffer((__gm__ int64_t*)ids);
 const int expert_slot=GetBlockIdx()%10, worker=GetBlockIdx()/10, expert=static_cast<int>(gi.GetValue(expert_slot));
 TPipe pipe;TBuf<TPosition::VECCALC> buffer;pipe.InitBuffer(buffer,65536);auto tmp=buffer.Get<uint8_t>();
 for(int part=0;part<2;++part) {
  const int rows=part?G:K,row_bytes=part?N*2:N/2,tile_rows=part?16:64;
  for(int begin=worker*tile_rows;begin<rows;begin+=4*tile_rows) {
   const int count=rows-begin<tile_rows?rows-begin:tile_rows;
   if(part)DataCopy(tmp,gs[(static_cast<int64_t>(expert)*rows+begin)*row_bytes],count*row_bytes);
   else DataCopy(tmp,gw[(static_cast<int64_t>(expert)*rows+begin)*row_bytes],count*row_bytes);
   SetFlag<HardEvent::MTE2_MTE3>(0);WaitFlag<HardEvent::MTE2_MTE3>(0);
   DataCopyParams copy{static_cast<uint16_t>(count),static_cast<uint16_t>(row_bytes/32),0,static_cast<uint16_t>(9*row_bytes/32)};
   if(part)DataCopy(ys[(static_cast<int64_t>(begin)*10+expert_slot)*row_bytes],tmp,copy);
   else DataCopy(yw[(static_cast<int64_t>(begin)*10+expert_slot)*row_bytes],tmp,copy);
   SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);
  }
 }
}
extern "C" int fnx_gather_pair(void*w,void*s,void*i,void*ow,void*os,int K,int G,int N,void*stream) {
 if(K<1||G<1||N!=1280)return -1;
 gather_pair<<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)i,(GM_ADDR)ow,(GM_ADDR)os,K,G,N);return 0;
}
