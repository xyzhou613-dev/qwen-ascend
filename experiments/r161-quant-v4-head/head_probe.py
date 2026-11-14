"""Real production mixer/head + argmax at M=1/2/4, no model or drafter."""
import hashlib,json,statistics,sys,time,urllib.request
from pathlib import Path
import torch,torch_npu
from fnx import engine as E
from fnx.load import load_nonlayer
from quant_head import QuantHead
P=Path(__file__).resolve().parent;device=int(sys.argv[1]);started=time.monotonic();transfers=None
torch.set_num_threads(1);torch.npu.set_device(0);torch.npu.set_per_process_memory_fraction(1/32,0)

def guard():
    global transfers
    h=json.load(urllib.request.urlopen('http://127.0.0.1:9002/health',timeout=5))
    assert h['service_revision']=='r152' and h['worker_alive']
    if transfers is None:transfers=h['pipeline_transfers']
    assert h['pipeline_transfers']==transfers,'PRODUCTION_REQUEST_PROGRESS_ABORT_HEAD_PROBE'
    assert torch.npu.mem_get_info()[0]>2*1024**3 and time.monotonic()-started<370

def err(a,b):
    a=a.cpu().float();b=b.cpu().float()
    return dict(relative_l2=float((a-b).norm()/a.norm().clamp_min(1e-12)),
        max_abs=float((a-b).abs().max()),finite=bool(torch.isfinite(b).all()))

@torch.no_grad()
def main():
    guard();embedding,head,mixer=load_nonlayer('npu:0','npu:0',with_lmhead=True)
    del embedding
    assert isinstance(head,E.DenseW8) and len(head.chunks)==4 and (head.K,head.N)==(2560,248320)
    quant=QuantHead.from_grouped(head,guard);torch.npu.synchronize();torch.npu.empty_cache()
    print('HEAD_PAIR_LOADED',quant.weight_fit_relative_l2,torch.npu.memory_allocated(),flush=True)
    records=[]
    for T in (1,2,4):
        guard();torch.manual_seed(16800+device+T)
        h=torch.empty((T,1,E.HCH),device='npu',dtype=torch.bfloat16)
        logits={k:torch.empty((T,head.N),device='npu',dtype=torch.bfloat16) for k in ('production','a8w8')}
        tokens={k:torch.empty(T,device='npu',dtype=torch.int64) for k in logits}
        def run(mode):
            mixed=E.gated_residual(h,mixer['hc_norm.weight'],mixer['input_mix_weight_down.weight'],
                mixer['input_mix_weight_up.weight'],None).view(T,E.H)
            logits[mode].copy_(head(mixed) if mode=='production' else quant(mixed))
            tokens[mode].copy_(logits[mode].argmax(-1))
        errors=[];batch_serial=[]
        for sample,amp in enumerate((.003,.03,.3,1.)):
            guard();h.copy_(torch.randn_like(h)*amp);run('production');run('a8w8')
            e=err(logits['production'],logits['a8w8']);assert e['finite']
            errors.append(dict(amplitude=amp,logit_error=e,
                argmax_equal_rows=int((tokens['production']==tokens['a8w8']).sum().cpu()),rows=T))
            if T>1:
                serial=[]
                for t in range(T):
                    mixed=E.gated_residual(h[t:t+1],mixer['hc_norm.weight'],
                        mixer['input_mix_weight_down.weight'],mixer['input_mix_weight_up.weight'],None).view(1,E.H)
                    serial.append(head(mixed).cpu())
                batch_serial.append(err(torch.cat(serial),logits['production']))
        stream=torch.npu.Stream();graphs={};samples={k:[] for k in logits};fresh={}
        with torch.npu.stream(stream):
            for mode in logits:
                run(mode);stream.synchronize();one=torch.npu.NPUGraph()
                with torch.npu.graph(one,stream=stream):run(mode)
                stream.synchronize()
                for _ in range(4):
                    guard();h.copy_(torch.randn_like(h)*.03);run(mode)
                    expected_logits=logits[mode].cpu().clone();expected_tokens=tokens[mode].cpu().clone()
                    one.replay();stream.synchronize()
                    assert torch.equal(logits[mode].cpu(),expected_logits) and torch.equal(tokens[mode].cpu(),expected_tokens)
                one.reset();fresh[mode]=True
            for mode in logits:
                run(mode);stream.synchronize();g=torch.npu.NPUGraph()
                with torch.npu.graph(g,stream=stream,pool=next(iter(graphs.values())).pool() if graphs else None):
                    for _ in range(8):run(mode)
                graphs[mode]=g;g.replay();stream.synchronize()
            for rep in range(12):
                guard()
                for mode in (tuple(logits) if rep%2==0 else tuple(reversed(logits))):
                    begin=torch.npu.Event(enable_timing=True);end=torch.npu.Event(enable_timing=True);begin.record(stream)
                    for _ in range(2):graphs[mode].replay()
                    end.record(stream);end.synchronize();samples[mode].append(begin.elapsed_time(end)/16)
            for g in graphs.values():g.reset()
        row=dict(T=T,median_ms={k:statistics.median(v) for k,v in samples.items()},samples_ms=samples,
            output_errors=errors,production_batch_vs_serial_logits=batch_serial,fresh_graph_verified=fresh)
        records.append(row);print('HEAD_COST',T,row['median_ms'],flush=True)
        (P/f'head-device{device}.json').write_text(json.dumps(dict(status='running',records=records),indent=2)+'\n')
    result=dict(status='passed_head_component',device=device,records=records,
        current_head_type=type(head).__name__,current_head_chunks=4,vocab=head.N,K=head.K,
        new_quantization='original grouped W8 reconstructed BF16 then per-output-channel W8; dynamic per-token A8; no calibration',
        weight_fit_relative_l2=quant.weight_fit_relative_l2,new_weight_storage_bytes=quant.weight_storage_bytes,
        peak_allocated_bytes=torch.npu.max_memory_allocated(),elapsed_s=time.monotonic()-started,
        allocator_fraction=1/32,exclusive=False,production_traffic_stable=True,
        scope='mixer+head+per-row argmax only; synthetic HC hidden; no NLL/task acceptance or full cycle')
    (P/f'head-device{device}.json').write_text(json.dumps(result,indent=2)+'\n')
    print('HEAD_PROBE_COMPLETE',device,result['elapsed_s'],flush=True)

if __name__=='__main__':main()
