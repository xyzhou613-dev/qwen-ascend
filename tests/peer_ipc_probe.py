"""Compare actual pipeline-sized BF16 IPC payloads, including pipe round trips."""
import json,os,statistics,subprocess,sys,tempfile,time,traceback
from pathlib import Path
from multiprocessing.connection import Connection
import multiprocessing as mp
import torch,torch_npu
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
try:
    from fnx.peer_ipc import Owner,Borrower,process_id
except ModuleNotFoundError as exc:
    if exc.name not in ('fnx','fnx.peer_ipc'):raise
    from peer_ipc import Owner,Borrower,process_id

WIDTH=10240;MAXIMUM=8192

def response(conn):
    if not conn.poll(90):raise TimeoutError('probe peer timeout')
    state,value=conn.recv()
    if state!='ok':raise RuntimeError(value)
    return value

def worker(fd,hostpath):
    conn=Connection(fd);peer=None
    try:
        torch.npu.set_device(0)
        dest=torch.empty(MAXIMUM,WIDTH,dtype=torch.bfloat16,device='npu:0')
        host=torch.from_file(hostpath,shared=True,size=2*MAXIMUM*WIDTH,dtype=torch.bfloat16).view(2,MAXIMUM,WIDTH)
        conn.send(('ok',process_id()))
        while True:
            op,args=conn.recv()
            if op=='init':peer=Borrower(args);value=None
            elif op=='copy':
                mode,slot,T,check=args
                if mode=='peer':peer.receive(dest[:T],slot)
                else:dest[:T].copy_(host[slot,:T]);torch.npu.synchronize()
                value=dest[:T].cpu().view(torch.int16).numpy().tobytes() if check else None
            elif op=='close':
                if peer is not None:peer.close();peer=None
                conn.send(('ok',None));break
            else:raise ValueError(op)
            conn.send(('ok',value))
    except Exception:
        conn.send(('error',traceback.format_exc()))
    finally:
        if peer is not None:peer.close()

def main():
    torch.npu.set_device(0)
    source=torch.empty(MAXIMUM,WIDTH,dtype=torch.bfloat16,device='npu:0')
    directory=Path(tempfile.mkdtemp(prefix='r91-probe-',dir='/root/ascend-dev'))
    hostpath=directory/'hidden.buf'
    with hostpath.open('wb') as f:f.truncate(2*MAXIMUM*WIDTH*2)
    host=torch.from_file(str(hostpath),shared=True,size=2*MAXIMUM*WIDTH,dtype=torch.bfloat16).view(2,MAXIMUM,WIDTH)
    conn,child=mp.Pipe();env=os.environ.copy();env['ASCEND_RT_VISIBLE_DEVICES']='1'
    proc=subprocess.Popen([sys.executable,__file__,'worker',str(child.fileno()),str(hostpath)],env=env,pass_fds=(child.fileno(),))
    child.close();owner=None;results=[]
    try:
        pid=response(conn)
        owner=Owner(MAXIMUM*WIDTH*2);owner.allow(pid)
        conn.send(('init',owner.descriptor()));response(conn)
        print('IPC_PEER_CONNECTED',flush=True)
        producer=torch.npu.Stream()
        with torch.npu.stream(producer):source[:128].fill_(3.5)
        owner.publish(source[:128],0)
        conn.send(('copy',('peer',0,128,True)))
        expected=torch.full((128,WIDTH),3.5,dtype=torch.bfloat16).view(torch.int16).numpy().tobytes()
        assert response(conn)==expected
        print('IPC_NONDEFAULT_STREAM_ACCEPT',flush=True)
        for slot in (0,1):
            source[:512].fill_(slot+2)
            owner.publish(source[:512],slot)
            conn.send(('copy',('peer',slot,512,True)))
        for slot in (0,1):
            expected=torch.full((512,WIDTH),slot+2,dtype=torch.bfloat16).view(torch.int16).numpy().tobytes()
            assert response(conn)==expected
        print('IPC_TWO_INFLIGHT_SLOTS_ACCEPT',flush=True)
        for T in (1,7,128,512,4096,8192):
            for turn in range(4):
                slot=turn%2
                # Include arbitrary BF16 bit patterns (NaNs too), compare bits.
                bits=((torch.arange(T*WIDTH,dtype=torch.int32)*17+turn*109+T)%65536-32768).to(torch.int16).view(T,WIDTH)
                source[:T].copy_(bits.view(torch.bfloat16));torch.npu.synchronize()
                owner.publish(source[:T],slot)
                conn.send(('copy',('peer',slot,T,True)))
                assert response(conn)==bits.numpy().tobytes(),(T,turn)
            print('IPC_BITEXACT',T,flush=True)
        for T in (1,128,512,4096,8192):
            row={'tokens':T,'bytes':T*WIDTH*2}
            for mode in ('host','peer','host','peer'):
                times=[]
                for i in range(24):
                    slot=i%2;t=time.perf_counter()
                    if mode=='peer':owner.publish(source[:T],slot)
                    else:host[slot,:T].copy_(source[:T]);torch.npu.synchronize()
                    conn.send(('copy',(mode,slot,T,False)));response(conn)
                    if i>=4:times.append((time.perf_counter()-t)*1000)
                row.setdefault(mode+'_ms',[]).append(statistics.median(times))
            results.append(row);print('ROUNDTRIP',json.dumps(row),flush=True)
        conn.send(('close',()));response(conn);proc.wait(timeout=15)
        owner.close();owner=None
        Path(os.environ.get('FNX_PEER_PROBE_OUTPUT','peer-probe-results.json')).write_text(json.dumps(results,indent=2))
        print('IPC_PROBE_ACCEPT',flush=True)
    finally:
        if proc.poll() is None:proc.terminate();proc.wait(timeout=15)
        if owner is not None:owner.close()
        conn.close();hostpath.unlink(missing_ok=True);directory.rmdir()

if __name__=='__main__':
    if len(sys.argv)>1:worker(int(sys.argv[2]),sys.argv[3])
    else:main()
