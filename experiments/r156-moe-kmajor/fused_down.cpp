// Included by w4_kmajor.cpp: one core owns 64 final output channels.
// Accumulate 320 K values per GM load, without transposing packed weights.
extern "C" __global__ __aicore__ void fused_down_merge(
 GM_ADDR weights,GM_ADDR scales,GM_ADDR ids,GM_ADDR input,GM_ADDR routing,
 GM_ADDR shared,GM_ADDR gate,GM_ADDR output) {
 constexpr int K=640,N=2560,T=64,C=320,G=10;
 const int col=GetBlockIdx()*T;
 GlobalTensor<int8_t> gw;GlobalTensor<int64_t> gi;GlobalTensor<bfloat16_t> gs,gx,gr,gsh,gg,go;
 gw.SetGlobalBuffer((__gm__ int8_t*)weights);gi.SetGlobalBuffer((__gm__ int64_t*)ids);
 gs.SetGlobalBuffer((__gm__ bfloat16_t*)scales);gx.SetGlobalBuffer((__gm__ bfloat16_t*)input);
 gr.SetGlobalBuffer((__gm__ bfloat16_t*)routing);gsh.SetGlobalBuffer((__gm__ bfloat16_t*)shared);
 gg.SetGlobalBuffer((__gm__ bfloat16_t*)gate);go.SetGlobalBuffer((__gm__ bfloat16_t*)output);
 TPipe pipe;TBuf<TPosition::VECCALC> qb,hb,fb,xb,xfb,brb,sb,sfb,ab,yb,wb,wfb,shb,gb,gfb,pb,rb,db;
 pipe.InitBuffer(qb,C*T/2);pipe.InitBuffer(hb,C*T*2);pipe.InitBuffer(fb,C*T*4);
 pipe.InitBuffer(xb,K*2);pipe.InitBuffer(xfb,K*4);pipe.InitBuffer(brb,K*8*4);
 pipe.InitBuffer(sb,5*T*2);pipe.InitBuffer(sfb,5*T*4);pipe.InitBuffer(ab,T*4);pipe.InitBuffer(yb,10*T*2);
 pipe.InitBuffer(wb,32);pipe.InitBuffer(wfb,64);pipe.InitBuffer(shb,T*2);pipe.InitBuffer(gb,32);
 pipe.InitBuffer(gfb,T*4);pipe.InitBuffer(pb,T*4);pipe.InitBuffer(rb,T*2);pipe.InitBuffer(db,T*4);
 auto q=qb.Get<int8_t>();auto h=hb.Get<half>();auto f=fb.Get<float>();auto x=xb.Get<bfloat16_t>();
 auto xf=xfb.Get<float>();auto br=brb.Get<float>();auto s=sb.Get<bfloat16_t>();auto sf=sfb.Get<float>();
 auto accum=ab.Get<float>();auto y=yb.Get<bfloat16_t>();auto w=wb.Get<bfloat16_t>();auto wf=wfb.Get<float>();
 auto sh=shb.Get<bfloat16_t>();auto g=gb.Get<bfloat16_t>();auto gf=gfb.Get<float>();
 auto p=pb.Get<float>();auto r=rb.Get<bfloat16_t>();auto d=db.Get<float>();
 for(int slot=0;slot<10;++slot) {
  const int expert=static_cast<int>(gi.GetValue(slot));
  DataCopy(x,gx[slot*K],K);SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  Cast(xf,x,RoundMode::CAST_NONE,K);PipeBarrier<PIPE_V>();Brcb(br,xf,K/8,{1,8});
  Duplicate(accum,0.0f,T);PipeBarrier<PIPE_V>();
  for(int chunk=0;chunk<2;++chunk) {
   DataCopyParams wp{C,1,(N-T)/64,0},sp{5,T*2/32,(N-T)*2/32,0};
   DataCopy(q,gw[(static_cast<int64_t>(expert)*K+chunk*C)*(N/2)+col/2],wp);
   DataCopy(s,gs[(static_cast<int64_t>(expert)*G+chunk*5)*N+col],sp);
   SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
   Cast(h,q.ReinterpretCast<int4b_t>(),RoundMode::CAST_NONE,C*T);PipeBarrier<PIPE_V>();
   Cast(f,h,RoundMode::CAST_NONE,C*T);Cast(sf,s,RoundMode::CAST_NONE,5*T);PipeBarrier<PIPE_V>();
   for(int first=0;first<C;first+=160)
    Mul(f[first*T],f[first*T],br[(chunk*C+first)*8],64,160,{1,1,0,8,8,1});
   PipeBarrier<PIPE_V>();
   for(int group=0;group<5;++group) {
    const int base=group*64*T;
    for(int rows=32;rows>0;rows/=2) {
     Add(f[base],f[base],f[base+rows*T],rows*T);PipeBarrier<PIPE_V>();
    }
    Mul(f[base],f[base],sf[group*T],T);PipeBarrier<PIPE_V>();
    Add(accum,accum,f[base],T);PipeBarrier<PIPE_V>();
   }
   SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
  }
  Cast(y[slot*T],accum,RoundMode::CAST_RINT,T);
  SetFlag<HardEvent::V_MTE2>(0);WaitFlag<HardEvent::V_MTE2>(0);
 }
 DataCopy(sh,gsh[col],T);
 DataCopyExtParams wp{1,20,0,0,0},gp{1,2,0,0,0};DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
 DataCopyPad(w,gr,wp,pad);DataCopyPad(g,gg,gp,pad);
 SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
 Cast(wf,w,RoundMode::CAST_NONE,G);Cast(gf,g,RoundMode::CAST_NONE,1);
 SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);float gate_value=gf.GetValue(0);
 Duplicate(accum,0.0f,T);PipeBarrier<PIPE_V>();
 // Preserve every BF16 rounding and the sequential sum from the production epilogue.
 for(int slot=0;slot<G;++slot) {
  Cast(xf,y[slot*T],RoundMode::CAST_NONE,T);PipeBarrier<PIPE_V>();
  Muls(xf,xf,wf.GetValue(slot),T);PipeBarrier<PIPE_V>();
  Cast(r,xf,RoundMode::CAST_RINT,T);PipeBarrier<PIPE_V>();Cast(p,r,RoundMode::CAST_NONE,T);PipeBarrier<PIPE_V>();
  Add(accum,accum,p,T);PipeBarrier<PIPE_V>();
 }
 Cast(r,accum,RoundMode::CAST_RINT,T);PipeBarrier<PIPE_V>();Cast(accum,r,RoundMode::CAST_NONE,T);
 Duplicate(gf,gate_value,T);PipeBarrier<PIPE_V>();Muls(d,gf,-1.0f,T);PipeBarrier<PIPE_V>();
 Exp(d,d,T);PipeBarrier<PIPE_V>();Adds(d,d,1.0f,T);Duplicate(gf,1.0f,T);PipeBarrier<PIPE_V>();
 Div(gf,gf,d,T);PipeBarrier<PIPE_V>();Cast(r,gf,RoundMode::CAST_RINT,T);PipeBarrier<PIPE_V>();
 Cast(gf,r,RoundMode::CAST_NONE,T);Cast(xf,sh,RoundMode::CAST_NONE,T);PipeBarrier<PIPE_V>();
 Mul(xf,xf,gf,T);PipeBarrier<PIPE_V>();Cast(r,xf,RoundMode::CAST_RINT,T);PipeBarrier<PIPE_V>();
 Cast(xf,r,RoundMode::CAST_NONE,T);PipeBarrier<PIPE_V>();Add(accum,accum,xf,T);PipeBarrier<PIPE_V>();
 Cast(r,accum,RoundMode::CAST_RINT,T);SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);
 DataCopy(go[col],r,T);
}
extern "C" int fnx_w4_fused(void*w,void*s,void*i,void*x,void*r,void*sh,void*g,void*y,void*stream) {
 fused_down_merge<<<40,nullptr,(aclrtStream)stream>>>((GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)i,(GM_ADDR)x,(GM_ADDR)r,(GM_ADDR)sh,(GM_ADDR)g,(GM_ADDR)y);return 0;
}
