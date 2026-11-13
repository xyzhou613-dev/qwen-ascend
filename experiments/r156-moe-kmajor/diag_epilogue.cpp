#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void moe_diagonal_epilogue(
 GM_ADDR cartesian,GM_ADDR weights,GM_ADDR shared,GM_ADDR gate,GM_ADDR output) {
 constexpr int W=2560,T=128,G=10;
 GlobalTensor<bfloat16_t> gy,gw,gs,gg,go;
 gy.SetGlobalBuffer((__gm__ bfloat16_t*)cartesian);gw.SetGlobalBuffer((__gm__ bfloat16_t*)weights);
 gs.SetGlobalBuffer((__gm__ bfloat16_t*)shared);gg.SetGlobalBuffer((__gm__ bfloat16_t*)gate);go.SetGlobalBuffer((__gm__ bfloat16_t*)output);
 TPipe pipe;TBuf<TPosition::VECCALC> yb,wb,sb,gb,xf,wf,pb,ab,rb,df,gr;
 pipe.InitBuffer(yb,G*T*2);pipe.InitBuffer(wb,32);pipe.InitBuffer(sb,T*2);pipe.InitBuffer(gb,32);
 pipe.InitBuffer(xf,T*4);pipe.InitBuffer(wf,64);pipe.InitBuffer(pb,T*4);pipe.InitBuffer(ab,T*4);
 pipe.InitBuffer(rb,T*2);pipe.InitBuffer(df,T*4);pipe.InitBuffer(gr,T*4);
 auto y=yb.Get<bfloat16_t>();auto w=wb.Get<bfloat16_t>();auto sh=sb.Get<bfloat16_t>();auto g=gb.Get<bfloat16_t>();
 auto x=xf.Get<float>();auto weights32=wf.Get<float>();auto p=pb.Get<float>();auto a=ab.Get<float>();
 auto r=rb.Get<bfloat16_t>();auto d=df.Get<float>();auto gate32=gr.Get<float>();
 const int col=GetBlockIdx()*T;
 for(int j=0;j<G;++j)DataCopy(y[j*T],gy[j*W+col],T);
 DataCopy(sh,gs[col],T);
 DataCopyExtParams wp{1,20,0,0,0},gp{1,2,0,0,0};DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};
 DataCopyPad(w,gw,wp,pad);DataCopyPad(g,gg,gp,pad);
 SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
 Cast(weights32,w,RoundMode::CAST_NONE,G);Cast(gate32,g,RoundMode::CAST_NONE,1);
 SetFlag<HardEvent::V_S>(0);WaitFlag<HardEvent::V_S>(0);float gate_value=gate32.GetValue(0);
 Duplicate(a,0.0f,T);PipeBarrier<PIPE_V>();
 for(int j=0;j<G;++j) {
  Cast(x,y[j*T],RoundMode::CAST_NONE,T);PipeBarrier<PIPE_V>();
  Muls(x,x,weights32.GetValue(j),T);PipeBarrier<PIPE_V>();
  Cast(r,x,RoundMode::CAST_RINT,T);PipeBarrier<PIPE_V>();Cast(p,r,RoundMode::CAST_NONE,T);PipeBarrier<PIPE_V>();
  Add(a,a,p,T);PipeBarrier<PIPE_V>();
 }
 Cast(r,a,RoundMode::CAST_RINT,T);PipeBarrier<PIPE_V>();Cast(a,r,RoundMode::CAST_NONE,T);
 Duplicate(gate32,gate_value,T);PipeBarrier<PIPE_V>();Muls(d,gate32,-1.0f,T);PipeBarrier<PIPE_V>();
 Exp(d,d,T);PipeBarrier<PIPE_V>();Adds(d,d,1.0f,T);Duplicate(gate32,1.0f,T);PipeBarrier<PIPE_V>();
 Div(gate32,gate32,d,T);PipeBarrier<PIPE_V>();Cast(r,gate32,RoundMode::CAST_RINT,T);PipeBarrier<PIPE_V>();
 Cast(gate32,r,RoundMode::CAST_NONE,T);Cast(x,sh,RoundMode::CAST_NONE,T);PipeBarrier<PIPE_V>();
 Mul(x,x,gate32,T);PipeBarrier<PIPE_V>();Cast(r,x,RoundMode::CAST_RINT,T);PipeBarrier<PIPE_V>();
 Cast(x,r,RoundMode::CAST_NONE,T);PipeBarrier<PIPE_V>();Add(a,a,x,T);PipeBarrier<PIPE_V>();Cast(r,a,RoundMode::CAST_RINT,T);
 SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(go[col],r,T);
}
extern "C" int fnx_diag_epilogue(void*y,void*w,void*s,void*g,void*o,void*stream) {
 moe_diagonal_epilogue<<<20,nullptr,(aclrtStream)stream>>>((GM_ADDR)y,(GM_ADDR)w,(GM_ADDR)s,(GM_ADDR)g,(GM_ADDR)o);return 0;
}
