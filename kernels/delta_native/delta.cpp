#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void delta_sequential(
    GM_ADDR state,GM_ADDR query,GM_ADDR key,GM_ADDR value,GM_ADDR beta,GM_ADDR decay,GM_ADDR output,
    int compact)
{
    constexpr uint32_t D=128,HEADS=48,AREA=D*D;
    GlobalTensor<float> gs,gq,gk,gv,gb,gg,go;
    gs.SetGlobalBuffer((__gm__ float*)state);gq.SetGlobalBuffer((__gm__ float*)query);
    gk.SetGlobalBuffer((__gm__ float*)key);gv.SetGlobalBuffer((__gm__ float*)value);
    gb.SetGlobalBuffer((__gm__ float*)beta);gg.SetGlobalBuffer((__gm__ float*)decay);
    go.SetGlobalBuffer((__gm__ float*)output);
    TPipe pipe;
    TBuf<TPosition::VECCALC> sb,qb,kb,vb,mb,db,tb,ob,rawb,rateb;
    pipe.InitBuffer(sb,AREA*4);pipe.InitBuffer(qb,D*4);pipe.InitBuffer(kb,D*4);
    pipe.InitBuffer(vb,D*4);pipe.InitBuffer(mb,D*4);pipe.InitBuffer(db,D*4);
    pipe.InitBuffer(tb,D*4);pipe.InitBuffer(ob,D*4);
    auto s=sb.Get<float>();auto q=qb.Get<float>();auto k=kb.Get<float>();auto v=vb.Get<float>();
    auto mem=mb.Get<float>();auto delta=db.Get<float>();auto tmp=tb.Get<float>();auto out=ob.Get<float>();
    GlobalTensor<bfloat16_t> cq, ck, cv, cb;
    cq.SetGlobalBuffer((__gm__ bfloat16_t*)query);
    ck.SetGlobalBuffer((__gm__ bfloat16_t*)key);
    cv.SetGlobalBuffer((__gm__ bfloat16_t*)value);
    cb.SetGlobalBuffer((__gm__ bfloat16_t*)beta);
    pipe.InitBuffer(rawb, (3 * D + 16) * sizeof(bfloat16_t));
    pipe.InitBuffer(rateb, 16 * sizeof(float));
    auto raw = rawb.Get<bfloat16_t>();
    auto rates = rateb.Get<float>();
    for(uint32_t head=GetBlockIdx();head<HEADS;head+=GetBlockNum()) {
        DataCopy(s,gs[head*AREA],AREA);
        float rate;
        if (compact) {
            // Three consecutive value heads share one query/key head.
            DataCopy(raw, cq[(head / 3) * D], D);
            DataCopy(raw[D], ck[(head / 3) * D], D);
            DataCopy(raw[2 * D], cv[head * D], D);
            DataCopy(raw[3 * D], cb[(head / 16) * 16], 16);
            SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
            Cast(q, raw, RoundMode::CAST_NONE, D);
            Cast(k, raw[D], RoundMode::CAST_NONE, D);
            Cast(v, raw[2 * D], RoundMode::CAST_NONE, D);
            Cast(rates, raw[3 * D], RoundMode::CAST_NONE, 16);
            SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);
            rate = rates.GetValue(head % 16);
        } else {
            DataCopy(q,gq[head*D],D);DataCopy(k,gk[head*D],D);DataCopy(v,gv[head*D],D);
            SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
            SetFlag<HardEvent::MTE2_S>(0);WaitFlag<HardEvent::MTE2_S>(0);
            rate = gb.GetValue(head);
        }
        const float gate = gg.GetValue(head);
        Muls(s,s,gate,AREA);PipeBarrier<PIPE_V>();
        Duplicate(mem,0.0f,D);PipeBarrier<PIPE_V>();
        for(uint32_t row=0;row<D;++row) {
            float scalar=k.GetValue(row);
            Muls(tmp,s[row*D],scalar,D);PipeBarrier<PIPE_V>();
            Add(mem,mem,tmp,D);PipeBarrier<PIPE_V>();
        }
        Sub(delta,v,mem,D);PipeBarrier<PIPE_V>();
        Muls(delta,delta,rate,D);PipeBarrier<PIPE_V>();
        Duplicate(out,0.0f,D);PipeBarrier<PIPE_V>();
        for(uint32_t row=0;row<D;++row) {
            float ks=k.GetValue(row),qs=q.GetValue(row);
            Muls(tmp,delta,ks,D);PipeBarrier<PIPE_V>();
            Add(s[row*D],s[row*D],tmp,D);PipeBarrier<PIPE_V>();
            Muls(tmp,s[row*D],qs,D);PipeBarrier<PIPE_V>();
            Add(out,out,tmp,D);PipeBarrier<PIPE_V>();
        }
        SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
        DataCopy(gs[head*AREA],s,AREA);DataCopy(go[head*D],out,D);
        SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);
        PipeBarrier<PIPE_ALL>();
    }
}

extern "C" int fnx_delta(void* state,void* query,void* key,void* value,
                          void* beta,void* decay,void* output,int blocks,void* stream) {
    if(blocks<1||blocks>40)return -1;
    delta_sequential<<<blocks,nullptr,(aclrtStream)stream>>>((GM_ADDR)state,(GM_ADDR)query,
        (GM_ADDR)key,(GM_ADDR)value,(GM_ADDR)beta,(GM_ADDR)decay,(GM_ADDR)output,0);
    return 0;
}

extern "C" int fnx_delta_compact(void* state, void* query, void* key, void* value,
                                 void* beta, void* decay, void* output, int blocks, void* stream) {
    if(blocks<1||blocks>40)return -1;
    delta_sequential<<<blocks,nullptr,(aclrtStream)stream>>>((GM_ADDR)state,(GM_ADDR)query,
        (GM_ADDR)key,(GM_ADDR)value,(GM_ADDR)beta,(GM_ADDR)decay,(GM_ADDR)output,1);
    return 0;
}
