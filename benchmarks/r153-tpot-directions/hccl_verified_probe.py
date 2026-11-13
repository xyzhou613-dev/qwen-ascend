import datetime, json, os, statistics, subprocess, sys, time, urllib.request
from pathlib import Path

P = Path(__file__).resolve().parent

def worker(rank):
    import torch, torch_npu
    import torch.distributed as dist
    started = time.monotonic()
    torch.set_num_threads(1)
    torch.npu.set_device(rank)
    torch.npu.set_per_process_memory_fraction(1/128, rank)
    cpus = set(range(32, 64)) if rank == 0 else set(range(0, 32))
    os.sched_setaffinity(0, cpus & os.sched_getaffinity(0))

    def guard():
        h = json.load(urllib.request.urlopen('http://127.0.0.1:9002/health', timeout=5))
        assert h['service_revision'] == 'r152' and h['worker_alive']
        assert torch.npu.mem_get_info()[0] > 1536*1024**2
        assert time.monotonic()-started < 180

    guard()
    dist.init_process_group('hccl', init_method='tcp://127.0.0.1:29853',
                            rank=rank, world_size=2, timeout=datetime.timedelta(seconds=45))
    results=[]
    for label, size, dtype in [('hidden_bf16',2560,torch.bfloat16),
                              ('hc_hidden_bf16',10240,torch.bfloat16),
                              ('hidden_fp32',2560,torch.float32)]:
        guard()
        x = torch.full((size,), rank+1., device=f'npu:{rank}', dtype=dtype)
        dist.all_reduce(x)
        torch.npu.synchronize()
        assert bool((x == 3).all().item())
        x.zero_()
        for _ in range(16): dist.all_reduce(x)
        torch.npu.synchronize()
        dist.barrier()
        eager=[]
        for _ in range(8):
            a=torch.npu.Event(enable_timing=True); b=torch.npu.Event(enable_timing=True)
            a.record()
            for _ in range(32): dist.all_reduce(x)
            b.record(); b.synchronize()
            eager.append(a.elapsed_time(b)*1000/32)
        dist.barrier()
        stream=torch.npu.Stream(device=rank)
        graph=torch.npu.NPUGraph()
        with torch.npu.stream(stream):
            dist.all_reduce(x); stream.synchronize()
            with torch.npu.graph(graph, stream=stream):
                for _ in range(96): dist.all_reduce(x)
            graph.replay(); stream.synchronize()
            # Replay must communicate fresh device data, not just captured zeros.
            # After 96 pairwise sums: (1+2)*2^-100 * 2^95 = 3/32.
            x.fill_((rank+1)*2.0**-100);stream.synchronize();dist.barrier()
            graph.replay();stream.synchronize()
            assert bool((x == 3/32).all().item()),'dynamic graph allreduce failed'
            x.zero_();stream.synchronize();dist.barrier()
            event_us=[]; host_ms=[]
            for _ in range(12):
                guard()
                a=torch.npu.Event(enable_timing=True); b=torch.npu.Event(enable_timing=True)
                t=time.perf_counter(); a.record(stream)
                for _ in range(2): graph.replay()
                b.record(stream); b.synchronize()
                host_ms.append((time.perf_counter()-t)*500)
                event_us.append(a.elapsed_time(b)*1000/192)
        graph.reset()
        dist.barrier()
        row=dict(name=label,bytes=size*x.element_size(),eager_us=eager,
                 dynamic_graph_sum_verified=True,
                 graph_us_per_allreduce=event_us,graph_host_ms_per_96=host_ms,
                 median_graph_us=statistics.median(event_us),
                 median_eager_us=statistics.median(eager))
        results.append(row)
        (P/f'hccl-rank{rank}.json').write_text(json.dumps(dict(status='running',rank=rank,records=results),indent=2))
        print('HCCL_RESULT',rank,label,row['median_graph_us'],row['median_eager_us'],flush=True)
    dist.destroy_process_group()
    (P/f'hccl-rank{rank}.json').write_text(json.dumps(dict(status='passed',rank=rank,records=results,
        elapsed_s=time.monotonic()-started,max_allocated_bytes=torch.npu.max_memory_allocated(),exclusive=False),indent=2))

def parent():
    env=os.environ.copy()
    env.update(json.load(open(P/'production-selected-config.json')))
    for key in ('RANK_TABLE_FILE','RANK_SIZE','RANK_ID','RANK','WORLD_SIZE','LOCAL_RANK','MASTER_ADDR','MASTER_PORT'):
        env.pop(key,None)
    env.update(ASCEND_RT_VISIBLE_DEVICES='0,1',HCCL_BUFFSIZE='8',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    handles=[]; processes=[]; deadline=time.monotonic()+210
    try:
        for rank in range(2):
            f=open(P/f'hccl-rank{rank}.log','w'); handles.append(f)
            processes.append(subprocess.Popen([sys.executable,__file__,str(rank)],env=env,stdout=f,stderr=subprocess.STDOUT))
        for child in processes:
            child.wait(timeout=max(1,deadline-time.monotonic()))
        assert all(c.returncode == 0 for c in processes), [c.returncode for c in processes]
    finally:
        for child in processes:
            if child.poll() is None: child.terminate()
        for child in processes:
            try: child.wait(timeout=8)
            except subprocess.TimeoutExpired: child.kill(); child.wait()
        for f in handles: f.close()

if __name__=='__main__':
    if len(sys.argv)>1: worker(int(sys.argv[1]))
    else: parent()
