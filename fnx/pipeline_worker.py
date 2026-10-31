"""One device per process; only ordered commands arrive over an inherited pipe."""
import ctypes
import os
import signal
import sys
import traceback
ctypes.CDLL(None).prctl(1,signal.SIGTERM)

import torch
from multiprocessing.connection import Connection
from . import engine as E
from .model import FnModel
from .index_buckets import capture_index_buckets,decode_graph,chunk_graph


@torch.no_grad()
def main():
    fd,P,split,n=map(int,sys.argv[1:5]);hpath,lpath=sys.argv[5:7]
    conn=Connection(fd)
    try:
        assert torch.npu.device_count()==1,'worker must expose exactly one device'
        torch.npu.set_device(0)
        maximum=E._CHUNK
        shared=torch.from_file(hpath,shared=True,size=2*maximum*E.HCH,dtype=torch.bfloat16).view(2,maximum,1,E.HCH)
        logits=torch.from_file(lpath,shared=True,size=E.VOCAB,dtype=torch.bfloat16)
        m=FnModel(P=P,lpc=n,dev0='npu:0',dev1='npu:0',nlayers=n,layer_start=split,
                  with_head=True,with_ngram=False,verbose=False)
        stage_input=torch.zeros(1,1,E.HCH,device='npu:0',dtype=torch.bfloat16)
        stage_chunk=torch.zeros(maximum,1,E.HCH,device='npu:0',dtype=torch.bfloat16)
        fused_head=os.environ.get('FNX_PIPELINE_FUSED_HEAD','0')=='1'
        greedy=torch.zeros((),dtype=torch.int64,device='npu:0')
        def body():
            h=stage_input
            for L in m.layers0:h=L.forward(h,m.pos0)
            m.h0.copy_(h)
            if fused_head:
                m._head_device()
                greedy.copy_(m.row_logits().argmax())
        def chunk_body():
            h=stage_chunk[:E._CHUNK]
            for L in m.layers0:h=L.forward_chunk(h,m.pos0,E._CHUNK)
            m.chunk_h0.copy_(h)
        m.body0=body;m._chunk_body0=chunk_body
        snapshots={};next_handle=0
        conn.send(('ok',{'pid':os.getpid(),'layers':len(m.layers0),'visible':os.environ.get('ASCEND_RT_VISIBLE_DEVICES')}))
        while True:
            op,args=conn.recv()
            if op=='capture':m.capture();value=None
            elif op=='capture_chunk':m.capture_chunk();capture_index_buckets(m);value=None
            elif op=='prefill':
                slot,T=args;pos=m._host_pos
                stage_chunk[:T].copy_(shared[slot,:T]);torch.npu.synchronize(0)
                short=pos+T<=2048
                bundle=chunk_graph(m,pos+T,T)
                if short or bundle is None:
                    h=stage_chunk[:T]
                    for L in m.layers0:
                        L.short_qsa=short and L.index_enabled
                        L.reference_short=L.idx<38
                        h=L.forward_chunk(h,pos,T)
                    for L in m.layers0:L.short_qsa=False
                else:bundle['gc'].replay()
                m.pos0.add_(T);m.pos1.add_(T);m._host_pos+=T
                torch.npu.synchronize(0);value=None
            elif op=='step':
                with_head,=args
                stage_input.copy_(shared[0,:1]);torch.npu.synchronize(0)
                graph=m.g0_short if m._host_pos<2048 and m.g0_short is not None else decode_graph(m,m._host_pos+1)
                if graph is not None:graph.replay()
                else:body()
                if with_head and not fused_head:m._head_device()
                m.pos0.add_(1);m.pos1.add_(1);m._host_pos+=1
                torch.npu.synchronize(0)
                value=int(greedy if fused_head else m.row_logits().argmax()) if with_head else None
            elif op=='sample':
                temperature,=args
                probs=torch.softmax(m.row_logits().float()/temperature,-1)
                value=int(torch.multinomial(probs,1))
            elif op=='logits':
                logits.copy_(m.row_logits());torch.npu.synchronize(0);value=None
            elif op=='reset':m.reset();value=None
            elif op=='save':
                to_host,compact=args
                st=m.save_state(to_host=to_host,compact=compact)
                handle=next_handle;next_handle+=1;snapshots[handle]=st
                value=(handle,m.state_bytes(st))
            elif op=='release':snapshots.pop(args[0]);value=None
            elif op=='load':m.load_state(snapshots[args[0]]);value=None
            elif op=='save_into':
                st=snapshots[args[0]];m.save_into(st);value=m.state_bytes(st)
            elif op=='to_host':
                handle=args[0];snapshots[handle]=m.state_to_host(snapshots[handle])
                value=m.state_bytes(snapshots[handle])
            else:raise ValueError('Unknown pipeline command '+op)
            conn.send(('ok',value))
            st=None  # snapshots owns cache tensors; do not retain the last save locally
    except (EOFError,BrokenPipeError):pass
    except Exception:
        error=traceback.format_exc();print(error,flush=True)
        try:conn.send(('error',error))
        except (EOFError,BrokenPipeError):pass

if __name__=='__main__':main()
