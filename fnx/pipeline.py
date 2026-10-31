"""Two single-device processes with bounded, double-buffered prefill IPC."""
import atexit
from collections import deque
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from . import engine as E
from .model import FnModel


class PipelineModel:
    def __init__(self, P=262144, lpc=24, nlayers=48, with_head=True, with_ngram=False, **kw):
        if E.B != 1 or with_ngram:
            raise ValueError('process pipeline requires batch=1 and NO_PLE/NO_NGRAM')
        assert torch.npu.device_count() == 1, 'parent must expose only pipeline device 0'
        self.P, self.n = P, nlayers or 48
        self.split = int(os.environ.get('FNX_PIPELINE_SPLIT', str(lpc)))
        self.maximum = E._CHUNK
        self.directory = Path(tempfile.mkdtemp(prefix='fnx-pp-', dir=os.environ.get('FNX_IPC_DIR','/root/ascend-dev')))
        self.hpath, self.lpath = self.directory/'hidden.buf', self.directory/'logits.buf'
        for path,size in ((self.hpath,2*self.maximum*E.HCH*2),(self.lpath,E.VOCAB*2)):
            with path.open('wb') as f:f.truncate(size)
            path.chmod(0o600)
        self.hidden = torch.from_file(str(self.hpath), shared=True, size=2*self.maximum*E.HCH,
                                      dtype=torch.bfloat16).view(2,self.maximum,1,E.HCH)
        self.remote_logits = torch.from_file(str(self.lpath), shared=True, size=E.VOCAB,dtype=torch.bfloat16)
        self.connection, peer = mp.Pipe()
        env=os.environ.copy()
        env['ASCEND_RT_VISIBLE_DEVICES']=env.get('FNX_PIPELINE_DEVICE1','1')
        env['FNX_GRAPH_G1']='0';env['FNX_GRAPH_HEAD']='0'
        self.worker_log=open(env.get('FNX_PIPELINE_LOG','/root/ascend-dev/logs/pipeline_worker.log'),'w')
        self.process=subprocess.Popen([sys.executable,'-m','fnx.pipeline_worker',str(peer.fileno()),
            str(P),str(self.split),str(self.n),str(self.hpath),str(self.lpath)],
            env=env,pass_fds=(peer.fileno(),),stdout=self.worker_log,stderr=subprocess.STDOUT)
        peer.close()
        self.pending=deque();self.slot=0;self.last_token=None
        atexit.register(self.close)
        self.stage0=FnModel(P=P,lpc=self.split,dev0='npu:0',dev1='npu:0',nlayers=self.split,
                            with_head=False,with_ngram=False,verbose=kw.get('verbose',False))
        self.worker_info=self._receive()
        print('[pipeline] worker ready',self.worker_info,flush=True)

    def __getattr__(self,name):
        return getattr(self.stage0,name)

    def _receive(self):
        if not self.connection.poll(1800):
            raise TimeoutError('pipeline worker did not respond')
        status,value=self.connection.recv()
        if status!='ok':raise RuntimeError('pipeline worker: '+str(value))
        return value

    def finish_prefill(self):
        while self.pending:
            self._receive();self.pending.popleft()

    def _rpc(self,op,*args):
        self.finish_prefill();self.connection.send((op,args));return self._receive()

    def capture(self):
        self.stage0.capture();self._rpc('capture')

    def capture_chunk(self):
        self.stage0.capture_chunk();self._rpc('capture_chunk')

    @torch.no_grad()
    def prefill_chunk(self,ids):
        offset=0
        while offset<len(ids):
            remaining=len(ids)-offset
            choices=[size for size in getattr(self.stage0,'chunk_graphs',{}) if size<=remaining]
            size=max(choices) if choices else min(remaining,self.maximum)
            if self.stage0._host_pos<2048 and size>512:size=min(512,remaining)
            self.stage0.prefill_chunk(ids[offset:offset+size])
            if len(self.pending)==2:
                self._receive();self.pending.popleft()
            self.hidden[self.slot,:size].copy_(self.stage0.prefill_hidden)
            torch.npu.synchronize(0)
            self.connection.send(('prefill',(self.slot,size)))
            self.pending.append(self.slot);self.slot^=1;offset+=size

    @torch.no_grad()
    def step_pad(self,tok,with_head=True):
        self.finish_prefill()
        self.stage0.step_pad(tok,with_head=False)
        self.hidden[0,:1].copy_(self.stage0.h0)
        torch.npu.synchronize(0)
        self.last_token=self._rpc('step',bool(with_head))

    def step(self,tok,with_head=True):
        self.step_pad(tok if isinstance(tok,int) else tok[0],with_head)

    def sample(self,temperature=0.):
        return int(self.last_token) if temperature<=0 else int(self._rpc('sample',float(temperature)))

    def row_logits(self,i=0):
        self._rpc('logits');return self.remote_logits.clone()

    def reset(self):
        self.finish_prefill();self.connection.send(('reset',()))
        self.stage0.reset();self._receive()

    def save_state(self,to_host=False,compact=None):
        self.finish_prefill()
        self.connection.send(('save',(to_host,compact)))
        local=self.stage0.save_state(to_host=to_host,compact=compact)
        handle,nbytes=self._receive()
        return {'stage0':local,'worker_handle':handle,'worker_bytes':nbytes}

    def load_state(self,st):
        self.finish_prefill();self.connection.send(('load',(st['worker_handle'],)))
        self.stage0.load_state(st['stage0']);self._receive()

    def save_into(self,st):
        self.finish_prefill();self.connection.send(('save_into',(st['worker_handle'],)))
        self.stage0.save_into(st['stage0']);st['worker_bytes']=self._receive()

    def state_bytes(self,st):
        return self.stage0.state_bytes(st['stage0'])+st['worker_bytes']

    def state_to_host(self,st):
        self.finish_prefill()
        self.connection.send(('to_host',(st['worker_handle'],)))
        st['stage0']=self.stage0.state_to_host(st['stage0'])
        st['worker_bytes']=self._receive()
        return st

    def close(self):
        process=self.__dict__.get('process')
        if process is not None and process.poll() is None:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
        conn=self.__dict__.get('connection')
        if conn is not None:conn.close()
        log=self.__dict__.get('worker_log')
        if log is not None:log.close()
        for path in (self.__dict__.get('hpath'),self.__dict__.get('lpath')):
            if path is not None:path.unlink(missing_ok=True)
        directory=self.__dict__.get('directory')
        if directory is not None:
            try:directory.rmdir()
            except OSError:pass
