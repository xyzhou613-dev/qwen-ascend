#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
namespace {constexpr int Groups=512, Tokens=2051, Pitch=2056;}

extern "C" __global__ __aicore__ void expand_groups(
    GM_ADDR chosen,GM_ADDR positions,GM_ADDR output,GM_ADDR gather_indices,
    GM_ADDR token_offsets,int rows,int capacity)
{
    GlobalTensor<float> source, result, offsets;
    GlobalTensor<uint32_t> gi;
    GlobalTensor<int64_t> pos;
    source.SetGlobalBuffer((__gm__ float*)chosen);
    result.SetGlobalBuffer((__gm__ float*)output);
    gi.SetGlobalBuffer((__gm__ uint32_t*)gather_indices);
    offsets.SetGlobalBuffer((__gm__ float*)token_offsets);
    pos.SetGlobalBuffer((__gm__ int64_t*)positions);
    TPipe pipe;
    TBuf<TPosition::VECCALC> srcb, idxb, sortedb, tempb, outb, gatherb, offsetsb;
    pipe.InitBuffer(srcb,Groups*4);pipe.InitBuffer(idxb,Groups*4);
    pipe.InitBuffer(sortedb,Groups*8);pipe.InitBuffer(tempb,Groups*32);
    pipe.InitBuffer(outb,Pitch*4);pipe.InitBuffer(gatherb,2048*4);pipe.InitBuffer(offsetsb,2048*4);
    auto values=srcb.Get<float>();auto ids=idxb.Get<uint32_t>();
    auto sorted=sortedb.Get<float>();auto tmp=tempb.Get<float>();
    auto out=outb.Get<float>();auto gather=gatherb.Get<uint32_t>();auto tail=offsetsb.Get<float>();
    CreateVecIndex(ids.ReinterpretCast<int32_t>(),static_cast<int32_t>(0),Groups);
    DataCopy(gather,gi,2048);DataCopy(tail,offsets,2048);
    SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
    for(int row=GetBlockIdx();row<rows;row+=GetBlockNum()) {
        DataCopy(values,source[row*Groups],Groups);
        SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
        Muls(values,values,-1.0f,Groups);PipeBarrier<PIPE_V>();
        Sort<float,true>(sorted,values,ids,tmp,16);PipeBarrier<PIPE_V>();
        Gather(out,sorted,gather,0,2048);PipeBarrier<PIPE_V>();
        Muls(out,out,-4.0f,2048);PipeBarrier<PIPE_V>();
        Add(out,out,tail,2048);PipeBarrier<PIPE_V>();
        const int64_t position=pos.GetValue(row);
        const int complete=(position+1)/4;
        const int valid=complete<Groups?complete:Groups;
        const int first=valid*4;
        const int aligned=(first+7)/8*8;
        if(aligned<Pitch)Duplicate(out[aligned],static_cast<float>(capacity),Pitch-aligned);
        SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
        for(int i=first;i<aligned;++i)out.SetValue(i,static_cast<float>(capacity));
        for(int i=0;i<3;++i) {
            const int token=complete*4+i;
            out.SetValue(first+i,token<=position?static_cast<float>(token):static_cast<float>(capacity));
        }
        SetFlag<HardEvent::S_MTE3>(0);WaitFlag<HardEvent::S_MTE3>(0);
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(result[row*Pitch],out,Pitch);
        SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);
        SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
    }
}

extern "C" int fnx_expand_groups(void* chosen,void* positions,void* output,
                                void* gather_indices,void* token_offsets,
                                int rows,int capacity,void* stream) {
    if(rows<1||capacity<2048)return -1;
    int blocks=rows<40?rows:40;
    expand_groups<<<blocks,nullptr,(aclrtStream)stream>>>((GM_ADDR)chosen,(GM_ADDR)positions,
        (GM_ADDR)output,(GM_ADDR)gather_indices,(GM_ADDR)token_offsets,rows,capacity);
    return 0;
}
