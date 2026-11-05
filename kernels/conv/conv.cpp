#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;

namespace {
constexpr int Rows = 128;
}

extern "C" __global__ __aicore__ void conv_four(
    GM_ADDR history, GM_ADDR input, GM_ADDR weights, GM_ADDR output,
    GM_ADDR buf_indices, GM_ADDR next_indices, GM_ADDR col_indices,
    int dimension, int order)
{
    GlobalTensor<bfloat16_t> gh, gx, gw, go;
    GlobalTensor<uint32_t> gi, gn, gc;
    gh.SetGlobalBuffer((__gm__ bfloat16_t*)history);
    gx.SetGlobalBuffer((__gm__ bfloat16_t*)input);
    gw.SetGlobalBuffer((__gm__ bfloat16_t*)weights);
    go.SetGlobalBuffer((__gm__ bfloat16_t*)output);
    gi.SetGlobalBuffer((__gm__ uint32_t*)buf_indices);
    gn.SetGlobalBuffer((__gm__ uint32_t*)next_indices);
    gc.SetGlobalBuffer((__gm__ uint32_t*)col_indices);
    TPipe pipe;
    TBuf<TPosition::VECCALC> srcb, weightb, packedb, nextb, productb, floatb, fwbuf, columnsb;
    TBuf<TPosition::VECCALC> ibuf, nbuf, cbuf;
    pipe.InitBuffer(srcb, Rows * 4 * 2);
    pipe.InitBuffer(weightb, Rows * 4 * 2);
    pipe.InitBuffer(packedb, Rows * 4 * 2);
    pipe.InitBuffer(nextb, Rows * 3 * 2);
    pipe.InitBuffer(productb, Rows * 4 * 2);
    pipe.InitBuffer(floatb, Rows * 4 * 4);
    pipe.InitBuffer(fwbuf, Rows * 4 * 4);
    pipe.InitBuffer(columnsb, Rows * 4 * 4);
    pipe.InitBuffer(ibuf, Rows * 4 * 4);
    pipe.InitBuffer(nbuf, Rows * 3 * 4);
    pipe.InitBuffer(cbuf, Rows * 4);
    auto src=srcb.Get<bfloat16_t>();auto weight=weightb.Get<bfloat16_t>();
    auto packed=packedb.Get<bfloat16_t>();auto next=nextb.Get<bfloat16_t>();
    auto rounded=productb.Get<bfloat16_t>();auto prod=floatb.Get<float>();
    auto wf=fwbuf.Get<float>();auto a=columnsb.Get<float>();
    auto b=a[Rows];auto c=a[Rows*2];auto d=a[Rows*3];
    auto idx=ibuf.Get<uint32_t>();auto ni=nbuf.Get<uint32_t>();auto ci=cbuf.Get<uint32_t>();
    DataCopy(idx,gi,Rows*4);DataCopy(ni,gn,Rows*3);DataCopy(ci,gc,Rows);
    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
    for (int begin=GetBlockIdx()*Rows;begin<dimension;begin+=GetBlockNum()*Rows) {
        DataCopy(src,gh[begin*3],Rows*3);
        DataCopy(src[Rows*3],gx[begin],Rows);
        DataCopy(weight,gw[begin*4],Rows*4);
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        Gather(packed.ReinterpretCast<half>(),src.ReinterpretCast<half>(),idx,0,Rows*4);
        Gather(next.ReinterpretCast<half>(),src.ReinterpretCast<half>(),ni,0,Rows*3);
        PipeBarrier<PIPE_V>();
        Cast(prod,packed,RoundMode::CAST_NONE,Rows*4);
        Cast(wf,weight,RoundMode::CAST_NONE,Rows*4);
        PipeBarrier<PIPE_V>();
        Mul(prod,prod,wf,Rows*4);PipeBarrier<PIPE_V>();
        // The reference rounds each BF16 product before FP32 accumulation.
        Cast(rounded,prod,RoundMode::CAST_RINT,Rows*4);PipeBarrier<PIPE_V>();
        Cast(prod,rounded,RoundMode::CAST_NONE,Rows*4);PipeBarrier<PIPE_V>();
        Gather(a,prod,ci,0,Rows);Gather(b,prod,ci,4,Rows);
        Gather(c,prod,ci,8,Rows);Gather(d,prod,ci,12,Rows);PipeBarrier<PIPE_V>();
        if(order==0) {
            Add(a,a,b,Rows);PipeBarrier<PIPE_V>();Add(a,a,c,Rows);
            PipeBarrier<PIPE_V>();Add(a,a,d,Rows);
        } else if(order==1) {
            Add(a,a,b,Rows);Add(c,c,d,Rows);PipeBarrier<PIPE_V>();Add(a,a,c,Rows);
        } else {
            Add(a,a,c,Rows);Add(b,b,d,Rows);PipeBarrier<PIPE_V>();Add(a,a,b,Rows);
        }
        PipeBarrier<PIPE_V>();
        Cast(rounded,a,RoundMode::CAST_RINT,Rows);
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(go[begin],rounded,Rows);
        DataCopy(gh[begin*3],next,Rows*3);
        SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);
        SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
    }
}

extern "C" int fnx_conv_four(void* history,void* input,void* weights,void* output,
                            void* buf_indices,void* next_indices,void* col_indices,
                            int dimension,int order,void* stream) {
    if(dimension<Rows||dimension%Rows||order<0||order>2)return -1;
    int blocks=dimension/Rows; if(blocks>40)blocks=40;
    conv_four<<<blocks,nullptr,(aclrtStream)stream>>>((GM_ADDR)history,(GM_ADDR)input,
        (GM_ADDR)weights,(GM_ADDR)output,(GM_ADDR)buf_indices,(GM_ADDR)next_indices,
        (GM_ADDR)col_indices,dimension,order);
    return 0;
}
