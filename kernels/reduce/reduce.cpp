#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
namespace { constexpr int MaxTile=2560; }

extern "C" __global__ __aicore__ void weighted_reduce(
    GM_ADDR input,GM_ADDR weights,GM_ADDR inverse,GM_ADDR output,
    int batch,int groups,int width,int row_stride,int mode)
{
    GlobalTensor<bfloat16_t> gx,gbw,gy;
    GlobalTensor<float> gw;
    GlobalTensor<int64_t> gi;
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)input);
    gbw.SetGlobalBuffer((__gm__ bfloat16_t*)weights);
    gw.SetGlobalBuffer((__gm__ float*)weights);
    gi.SetGlobalBuffer((__gm__ int64_t*)inverse);
    gy.SetGlobalBuffer((__gm__ bfloat16_t*)output);
    TPipe pipe;
    TBuf<TPosition::VECCALC> xb,wb,fb,wfb,rb,pb,ab;
    pipe.InitBuffer(xb,10*MaxTile*2);pipe.InitBuffer(wb,4*MaxTile*2);
    pipe.InitBuffer(fb,MaxTile*4);pipe.InitBuffer(wfb,MaxTile*4);
    pipe.InitBuffer(rb,MaxTile*2);pipe.InitBuffer(pb,MaxTile*4);
    pipe.InitBuffer(ab,MaxTile*4);
    auto x=xb.Get<bfloat16_t>();auto w=wb.Get<bfloat16_t>();
    auto xf=fb.Get<float>();auto wf=wfb.Get<float>();
    auto rounded=rb.Get<bfloat16_t>();auto product=pb.Get<float>();auto accum=ab.Get<float>();
    const int tile=batch<128?128:MaxTile;
    const int columns=width/tile;
    const int tasks=batch*columns;
    for(int task=GetBlockIdx();task<tasks;task+=GetBlockNum()) {
        const int token=task/columns, col=(task%columns)*tile;
        for(int group=0;group<groups;++group) {
            const int row=mode==2?static_cast<int>(gi.GetValue(token*groups+group)):token*groups+group;
            DataCopy(x[group*MaxTile],gx[static_cast<int64_t>(row)*row_stride+col],tile);
            if(mode==0)DataCopy(w[group*MaxTile],gbw[(static_cast<int64_t>(token)*groups+group)*width+col],tile);
        }
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        Duplicate(accum,0.0f,tile);PipeBarrier<PIPE_V>();
        for(int group=0;group<groups;++group) {
            Cast(xf,x[group*MaxTile],RoundMode::CAST_NONE,tile);
            if(mode==0) {
                Cast(wf,w[group*MaxTile],RoundMode::CAST_NONE,tile);PipeBarrier<PIPE_V>();
                Mul(xf,xf,wf,tile);
            } else {
                PipeBarrier<PIPE_V>();
                Muls(xf,xf,gw.GetValue(token*groups+group),tile);
            }
            PipeBarrier<PIPE_V>();
            Cast(rounded,xf,RoundMode::CAST_RINT,tile);PipeBarrier<PIPE_V>();
            Cast(product,rounded,RoundMode::CAST_NONE,tile);PipeBarrier<PIPE_V>();
            Add(accum,accum,product,tile);PipeBarrier<PIPE_V>();
        }
        if(mode==0) {Muls(accum,accum,0.25f,tile);PipeBarrier<PIPE_V>();}
        Cast(rounded,accum,RoundMode::CAST_RINT,tile);
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(gy[static_cast<int64_t>(token)*width+col],rounded,tile);
        SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
        SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);
    }
}

extern "C" int fnx_weighted_reduce(void* input,void* weights,void* inverse,void* output,
                                 int batch,int groups,int width,int row_stride,int mode,int order,void* stream) {
    if(batch<1||width!=2560||groups<1||groups>10||mode<0||mode>2||order!=0)return -1;
    const int tile=batch<128?128:MaxTile;
    int blocks=batch*(width/tile);if(blocks>40)blocks=40;
    weighted_reduce<<<blocks,nullptr,(aclrtStream)stream>>>((GM_ADDR)input,(GM_ADDR)weights,
        (GM_ADDR)inverse,(GM_ADDR)output,batch,groups,width,row_stride,mode);
    return 0;
}
