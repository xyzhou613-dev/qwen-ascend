"""Modify only a copied pipeline; no math, ordering or production source changes."""
from pathlib import Path
P=Path(__file__).resolve().parent
p=P/'fnx/pipeline.py';s=p.read_text()
s=s.replace('import torch\n','import torch\nfrom profile_support import span\n',1)
a=s.index('    @torch.no_grad()\n    def step_pad');b=s.index('\n    def step(',a)
s=s[:a]+'''    @torch.no_grad()
    def step_pad(self,tok,with_head=True):
        import profile_support as ps
        self.finish_prefill()
        with span('stage0_prepare_submit'):
            self.stage0.g0=decode_graph(self.stage0,self.stage0._host_pos+1)
            self.stage0.step_pad(tok,with_head=False)
        with span('stage0_transfer_wait'):
            if self.transport=='peer':
                self.peer_owner.publish(self.stage0.h0,0)
                self.transport_stats['peer_transfers']+=1
                self.transport_stats['peer_bytes']+=E.HCH*2
            else:
                self.hidden[0,:1].copy_(self.stage0.h0)
                torch.npu.synchronize(0)
                self.transport_stats['host_transfers']+=1
                self.transport_stats['host_bytes']+=E.HCH*2
        self.finish_prefill()
        with span('pipe_send'):
            self.connection.send(('step',(ps._seq,bool(with_head))))
        with span('pipe_recv_wait'):
            self.last_token=self._receive()
''' +s[b:];p.write_text(s)
p=P/'fnx/pipeline_worker.py';s=p.read_text()
s=s.replace('import torch\n','import torch\nimport time\nimport profile_support as ps\nfrom profile_support import span,make_profiler\n',1)
anchor="            if op=='peer_init':"
assert s.count(anchor)==1
s=s.replace(anchor,"""            if op=='clock':value=(time.time_ns(),time.perf_counter_ns())
            elif op=='prof_start':
                path,metric,tag=args
                profiler=make_profiler(path,metric);profiler.start();ps.begin(tag,True);value=None
            elif op=='prof_stop':
                path,=args;ps.finish(path);profiler.stop();profiler=None;value=None
            elif op=='host_start':ps.begin(args[0],False);value=None
            elif op=='host_stop':value=ps.finish(args[0])
            elif op=='peer_init':""",1)
a=s.index("            elif op=='step':");b=s.index("            elif op=='sample':",a)
s=s[:a]+'''            elif op=='step':
                seq,with_head=args;ps.set_seq(seq)
                with span('stage1_input_wait'):
                    if transport=='peer':peer.receive(stage_input,0)
                    else:stage_input.copy_(shared[0,:1]);torch.npu.synchronize(0)
                with span('stage1_graph_submit'):
                    graph=m.g0_short if m._host_pos<2048 and m.g0_short is not None else decode_graph(m,m._host_pos+1)
                    if graph is not None:graph.replay()
                    else:body()
                    if with_head and not fused_head:m._head_device()
                    m.pos0.add_(1);m.pos1.add_(1);m._host_pos+=1
                with span('stage1_greedy_wait'):
                    if with_head:value=int(greedy if fused_head else m.row_logits().argmax())
                    else:torch.npu.synchronize(0);value=None
''' +s[b:]
s=s.replace("            conn.send(('ok',value))\n            st=None", "            if op=='step':\n                with span('pipe_response_send'):conn.send(('ok',value))\n            else:conn.send(('ok',value))\n            st=None")
p.write_text(s)
for p in (P/'fnx/pipeline.py',P/'fnx/pipeline_worker.py'):compile(p.read_text(),str(p),'exec')
print('DIAGNOSTIC_PIPELINE_PATCHED',flush=True)
