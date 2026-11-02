#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;

// Fixed production hidden width and top-k; no quantization or router change.
extern "C" __global__ __aicore__ void combine_bf16(
    GM_ADDR source, GM_ADDR inverse, GM_ADDR weights, GM_ADDR output, uint32_t tokens)
{
    constexpr uint32_t H=2560, K=10;
    GlobalTensor<bfloat16_t> src, dst;
    GlobalTensor<int32_t> idx;
    GlobalTensor<float> scale;
    src.SetGlobalBuffer((__gm__ bfloat16_t*)source);
    dst.SetGlobalBuffer((__gm__ bfloat16_t*)output);
    idx.SetGlobalBuffer((__gm__ int32_t*)inverse);
    scale.SetGlobalBuffer((__gm__ float*)weights);
    TPipe pipe;
    TBuf<TPosition::VECCALC> rawBuffer, fpBuffer, roundBuffer, accBuffer;
    pipe.InitBuffer(rawBuffer,H*sizeof(bfloat16_t));
    pipe.InitBuffer(fpBuffer,H*sizeof(float));
    pipe.InitBuffer(roundBuffer,H*sizeof(bfloat16_t));
    pipe.InitBuffer(accBuffer,H*sizeof(float));
    auto raw=rawBuffer.Get<bfloat16_t>();
    auto fp=fpBuffer.Get<float>();
    auto rounded=roundBuffer.Get<bfloat16_t>();
    auto acc=accBuffer.Get<float>();
    for(uint32_t token=GetBlockIdx();token<tokens;token+=GetBlockNum()) {
        Duplicate(acc,0.0f,H);
        PipeBarrier<PIPE_V>();
        for(uint32_t expert=0;expert<K;++expert) {
            uint32_t position=token*K+expert;
            int32_t row=idx.GetValue(position);
            float probability=scale.GetValue(position);
            DataCopy(raw,src[(uint64_t)row*H],H);
            SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
            Cast(fp,raw,RoundMode::CAST_NONE,H);PipeBarrier<PIPE_V>();
            Muls(fp,fp,probability,H);PipeBarrier<PIPE_V>();
            Cast(rounded,fp,RoundMode::CAST_RINT,H);PipeBarrier<PIPE_V>();
            Cast(fp,rounded,RoundMode::CAST_NONE,H);PipeBarrier<PIPE_V>();
            Add(acc,acc,fp,H);PipeBarrier<PIPE_V>();
            SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
        }
        Cast(raw,acc,RoundMode::CAST_RINT,H);PipeBarrier<PIPE_V>();
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(dst[(uint64_t)token*H],raw,H);
        SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);
    }
}

extern "C" int fnx_combine(void* source,void* inverse,void* weights,void* output,
                            int tokens,int blocks,void* stream) {
    if(tokens<1||blocks<1||blocks>40)return -1;
    combine_bf16<<<blocks,nullptr,(aclrtStream)stream>>>(
        (GM_ADDR)source,(GM_ADDR)inverse,(GM_ADDR)weights,(GM_ADDR)output,(uint32_t)tokens);
    return 0;
}
