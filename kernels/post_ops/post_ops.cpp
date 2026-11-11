#include "kernel_operator.h"
#include "acl/acl.h"
using namespace AscendC;
extern "C" __global__ __aicore__ void hc_activation(GM_ADDR input,GM_ADDR mixed,GM_ADDR gates,int batch,int mode) {
 GlobalTensor<bfloat16_t> gx,gm,gg;gx.SetGlobalBuffer((__gm__ bfloat16_t*)input);gm.SetGlobalBuffer((__gm__ bfloat16_t*)mixed);gg.SetGlobalBuffer((__gm__ bfloat16_t*)gates);
 TPipe pipe;TBuf<TPosition::VECCALC> bb,vb,db,sb,ab,ob,gb;
 pipe.InitBuffer(bb,672);pipe.InitBuffer(vb,1344);pipe.InitBuffer(db,1344);pipe.InitBuffer(sb,1344);pipe.InitBuffer(ab,1344);pipe.InitBuffer(ob,640);pipe.InitBuffer(gb,32);
 auto b=bb.Get<bfloat16_t>();auto v=vb.Get<float>();auto d=db.Get<float>();auto s=sb.Get<float>();auto a=ab.Get<float>();auto out=ob.Get<bfloat16_t>();auto g=gb.Get<bfloat16_t>();
 DataCopyExtParams load{1,648,0,0,0};DataCopyPadExtParams<bfloat16_t> pad{false,0,0,0};DataCopyExtParams save{1,8,0,0,0};
 for(int token=GetBlockIdx();token<batch;token+=GetBlockNum()) {
  DataCopyPad(b,gx[static_cast<int64_t>(token)*324],load,pad);SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  Cast(v,b,RoundMode::CAST_NONE,324);PipeBarrier<PIPE_V>();Muls(v,v,0.25f,324);PipeBarrier<PIPE_V>();
  Cast(b,v,RoundMode::CAST_RINT,324);PipeBarrier<PIPE_V>();Cast(v,b,RoundMode::CAST_NONE,324);PipeBarrier<PIPE_V>();
  Muls(d,v,-1.0f,324);PipeBarrier<PIPE_V>();Exp(d,d,324);PipeBarrier<PIPE_V>();Adds(d,d,1.0f,324);PipeBarrier<PIPE_V>();
  if(mode&1)Reciprocal(s,d,324);
  else {Duplicate(a,1.0f,324);PipeBarrier<PIPE_V>();Div(s,a,d,324);}
  PipeBarrier<PIPE_V>();
  if(mode&2)Mul(a,v,s,320);else Div(a,v,d,320);
  PipeBarrier<PIPE_V>();Cast(out,a,RoundMode::CAST_RINT,320);Cast(g,s[320],RoundMode::CAST_RINT,4);PipeBarrier<PIPE_V>();
  Cast(a,g,RoundMode::CAST_NONE,4);PipeBarrier<PIPE_V>();Muls(a,a,2.0f,4);PipeBarrier<PIPE_V>();Cast(g,a,RoundMode::CAST_RINT,4);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(gm[static_cast<int64_t>(token)*320],out,320);DataCopyPad(gg[token*4],g,save);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" int fnx_hc_activation(void*x,void*m,void*g,int batch,int mode,void*stream) {
 if(batch<1||mode<0||mode>3)return -1;int blocks=batch<40?batch:40;
 hc_activation<<<blocks,nullptr,(aclrtStream)stream>>>((GM_ADDR)x,(GM_ADDR)m,(GM_ADDR)g,batch,mode);return 0;
}

extern "C" __global__ __aicore__ void hc_sigmoid_reduce(GM_ADDR input,GM_ADDR logits,GM_ADDR output,int batch) {
 GlobalTensor<bfloat16_t> gx,gl,gy;gx.SetGlobalBuffer((__gm__ bfloat16_t*)input);gl.SetGlobalBuffer((__gm__ bfloat16_t*)logits);gy.SetGlobalBuffer((__gm__ bfloat16_t*)output);
 TPipe pipe;TBuf<TPosition::VECCALC> xb,lb,fb,db,sb,rb,ab;
 pipe.InitBuffer(xb,1024);pipe.InitBuffer(lb,1024);pipe.InitBuffer(fb,2048);pipe.InitBuffer(db,2048);pipe.InitBuffer(sb,2048);pipe.InitBuffer(rb,1024);pipe.InitBuffer(ab,512);
 auto x=xb.Get<bfloat16_t>();auto l=lb.Get<bfloat16_t>();auto f=fb.Get<float>();auto d=db.Get<float>();auto s=sb.Get<float>();auto r=rb.Get<bfloat16_t>();auto a=ab.Get<float>();
 DataCopyParams load{4,8,152,0};
 for(int task=GetBlockIdx();task<batch*20;task+=GetBlockNum()) {
  const int token=task/20,col=(task%20)*128;const int64_t offset=static_cast<int64_t>(token)*10240+col;
  DataCopy(x,gx[offset],load);DataCopy(l,gl[offset],load);SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  Cast(f,l,RoundMode::CAST_NONE,512);PipeBarrier<PIPE_V>();Muls(d,f,-1.0f,512);PipeBarrier<PIPE_V>();Exp(d,d,512);PipeBarrier<PIPE_V>();Adds(d,d,1.0f,512);
  Duplicate(s,1.0f,512);PipeBarrier<PIPE_V>();Div(f,s,d,512);PipeBarrier<PIPE_V>();Cast(r,f,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();
  Cast(s,r,RoundMode::CAST_NONE,512);Cast(f,x,RoundMode::CAST_NONE,512);PipeBarrier<PIPE_V>();Mul(f,f,s,512);PipeBarrier<PIPE_V>();
  Cast(r,f,RoundMode::CAST_RINT,512);PipeBarrier<PIPE_V>();Cast(f,r,RoundMode::CAST_NONE,512);PipeBarrier<PIPE_V>();
  Duplicate(a,0.0f,128);PipeBarrier<PIPE_V>();
  for(int j=0;j<4;++j){Add(a,a,f[j*128],128);PipeBarrier<PIPE_V>();}
  Muls(a,a,0.25f,128);PipeBarrier<PIPE_V>();Cast(r,a,RoundMode::CAST_RINT,128);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(gy[static_cast<int64_t>(token)*2560+col],r,128);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" int fnx_hc_sigmoid_reduce(void*x,void*l,void*y,int batch,void*stream) {
 if(batch<1)return -1;int blocks=batch*20;if(blocks>40)blocks=40;
 hc_sigmoid_reduce<<<blocks,nullptr,(aclrtStream)stream>>>((GM_ADDR)x,(GM_ADDR)l,(GM_ADDR)y,batch);return 0;
}

extern "C" __global__ __aicore__ void post_mul(GM_ADDR input,GM_ADDR weight,GM_ADDR output,int n,int mode) {
 GlobalTensor<float> gx,gw;GlobalTensor<bfloat16_t> gy;gx.SetGlobalBuffer((__gm__ float*)input);gw.SetGlobalBuffer((__gm__ float*)weight);gy.SetGlobalBuffer((__gm__ bfloat16_t*)output);
 TPipe pipe;TBuf<TPosition::VECCALC> xb,wb,rb;pipe.InitBuffer(xb,1024);pipe.InitBuffer(wb,1024);pipe.InitBuffer(rb,512);
 auto x=xb.Get<float>();auto w=wb.Get<float>();auto r=rb.Get<bfloat16_t>();
 for(int tile=GetBlockIdx();tile<n/256;tile+=GetBlockNum()) {
  const int offset=tile*256;DataCopy(x,gx[offset],256);DataCopy(w,gw[offset],256);SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);
  if(mode){Cast(r,x,RoundMode::CAST_RINT,256);PipeBarrier<PIPE_V>();Cast(x,r,RoundMode::CAST_NONE,256);PipeBarrier<PIPE_V>();}
  Mul(x,x,w,256);PipeBarrier<PIPE_V>();Cast(r,x,RoundMode::CAST_RINT,256);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(gy[offset],r,256);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" int fnx_post_mul(void*x,void*w,void*y,int n,int mode,void*stream) {
 if((n!=6144&&n!=10240)||mode<0||mode>1)return -1;int blocks=n/256;if(blocks>40)blocks=40;
 post_mul<<<blocks,nullptr,(aclrtStream)stream>>>((GM_ADDR)x,(GM_ADDR)w,(GM_ADDR)y,n,mode);return 0;
}

extern "C" __global__ __aicore__ void moe_merge(GM_ADDR experts,GM_ADDR shared,GM_ADDR gate,GM_ADDR output,int batch) {
 GlobalTensor<bfloat16_t> ge,gs,go;GlobalTensor<uint16_t> gg;
 ge.SetGlobalBuffer((__gm__ bfloat16_t*)experts);gs.SetGlobalBuffer((__gm__ bfloat16_t*)shared);go.SetGlobalBuffer((__gm__ bfloat16_t*)output);gg.SetGlobalBuffer((__gm__ uint16_t*)gate);
 TPipe pipe;TBuf<TPosition::VECCALC> eb,sb,rb,ef,sf,gf,df;
 pipe.InitBuffer(eb,256);pipe.InitBuffer(sb,256);pipe.InitBuffer(rb,256);pipe.InitBuffer(ef,512);pipe.InitBuffer(sf,512);pipe.InitBuffer(gf,512);pipe.InitBuffer(df,512);
 auto e=eb.Get<bfloat16_t>();auto s=sb.Get<bfloat16_t>();auto r=rb.Get<bfloat16_t>();auto e32=ef.Get<float>();auto s32=sf.Get<float>();auto g32=gf.Get<float>();auto d=df.Get<float>();
 for(int task=GetBlockIdx();task<batch*20;task+=GetBlockNum()) {
  const int token=task/20,col=(task%20)*128;const int64_t offset=static_cast<int64_t>(token)*2560+col;
  DataCopy(e,ge[offset],128);DataCopy(s,gs[offset],128);
  union {uint32_t bits;float value;} u;u.bits=static_cast<uint32_t>(gg.GetValue(token))<<16;
  Duplicate(g32,u.value,128);PipeBarrier<PIPE_V>();Muls(d,g32,-1.0f,128);PipeBarrier<PIPE_V>();Exp(d,d,128);PipeBarrier<PIPE_V>();Adds(d,d,1.0f,128);
  Duplicate(g32,1.0f,128);PipeBarrier<PIPE_V>();Div(g32,g32,d,128);PipeBarrier<PIPE_V>();Cast(r,g32,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();Cast(g32,r,RoundMode::CAST_NONE,128);
  SetFlag<HardEvent::MTE2_V>(0);WaitFlag<HardEvent::MTE2_V>(0);Cast(e32,e,RoundMode::CAST_NONE,128);Cast(s32,s,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
  Mul(s32,s32,g32,128);PipeBarrier<PIPE_V>();Cast(r,s32,RoundMode::CAST_RINT,128);PipeBarrier<PIPE_V>();Cast(s32,r,RoundMode::CAST_NONE,128);PipeBarrier<PIPE_V>();
  Add(e32,e32,s32,128);PipeBarrier<PIPE_V>();Cast(r,e32,RoundMode::CAST_RINT,128);
  SetFlag<HardEvent::V_MTE3>(0);WaitFlag<HardEvent::V_MTE3>(0);DataCopy(go[offset],r,128);
  SetFlag<HardEvent::MTE3_MTE2>(0);WaitFlag<HardEvent::MTE3_MTE2>(0);SetFlag<HardEvent::MTE3_V>(0);WaitFlag<HardEvent::MTE3_V>(0);
 }
}
extern "C" int fnx_moe_merge(void*e,void*s,void*g,void*y,int batch,void*stream) {
 if(batch<1)return -1;int blocks=batch*20;if(blocks>40)blocks=40;
 moe_merge<<<blocks,nullptr,(aclrtStream)stream>>>((GM_ADDR)e,(GM_ADDR)s,(GM_ADDR)g,(GM_ADDR)y,batch);return 0;
}
