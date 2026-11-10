#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void scalar_nop(GM_ADDR out) {
 GlobalTensor<int32_t> y;y.SetGlobalBuffer((__gm__ int32_t*)out);y.SetValue(GetBlockIdx(),1);
}
extern "C" int fnx_nop(void* out,int blocks,void* stream) {
 scalar_nop<<<blocks,nullptr,(aclrtStream)stream>>>((GM_ADDR)out);return 0;
}
