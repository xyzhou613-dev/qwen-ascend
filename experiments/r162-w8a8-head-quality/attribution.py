"""Optional post-completion counterfactual on the failed zh/65536 slice.

Never changes the frozen model, quality client, threshold, or absolute deadline.
Only pauses the parent shell after all 1200 task results and summary exist.
The original independent timer/guardian remain in charge of restoration.
"""
import hashlib,json,math,os,signal,subprocess,time,urllib.request
from pathlib import Path
P=Path(__file__).resolve().parent;start=int((P/'window-start.txt').read_text())
def state(unit):return subprocess.check_output(['systemctl','show',unit,'-p','ActiveState','--value'],text=True).strip()
def status(stage,**extra):
    row=dict(epoch=time.time(),stage=stage,**extra)
    (P/'attribution-status.json').write_text(json.dumps(row,indent=2)+'\n');print(stage,flush=True)
def counts():
    return sum(x.endswith(b'\n') for x in (P/'quality-combined.jsonl').read_bytes().splitlines(keepends=True))
def post(route,body,timeout=240):
    req=urllib.request.Request('http://127.0.0.1:19002'+route,data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})
    return json.load(urllib.request.urlopen(req,timeout=timeout))
def container(script,timeout):
    return subprocess.run(['docker','exec','bole-model-Flash','/root/ascend-dev/venv/bin/python',str(P/script)],check=True,timeout=timeout)
def stop_signal(*_):raise SystemExit('ATTRIBUTION_SERVICE_CANCELLED')
signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
assert not (P/'attribution-pause.json').exists()
frozen=json.loads((P/'attribution-harness-manifest.json').read_text())
assert all(hashlib.sha256((P/n).read_bytes()).hexdigest()==h for n,h in frozen.items())
status('waiting_for_all_1200_tasks')
paused=False
try:
    next_service_check=0
    while True:
        if not (P/'lease.active').exists():
            status('skipped_main_already_restoring');break
        if time.monotonic()>=next_service_check:
            if state('fnx-r162-run.service')!='active' or state('fnx-r162-rescue.service')!='inactive':
                status('skipped_main_already_restoring');break
            next_service_check=time.monotonic()+5
        # No additional phase near the original recovery reserve.
        if time.time()>=start+16800:status('skipped_insufficient_original_window');break
        summary=P/'quality-combined-summary.json'
        if summary.exists() and counts()==1200 and json.loads(summary.read_text())['completed']==1200:
            if state('fnx-r162-run.service')!='active' or state('fnx-r162-rescue.service')!='inactive':
                status('skipped_main_already_restoring');break
            pid=int(subprocess.check_output(['systemctl','show','fnx-r162-run.service','-p','MainPID','--value'],text=True))
            proc=Path('/proc',str(pid));argv=proc.joinpath('cmdline').read_bytes().split(b'\0')
            assert str(P/'controller.sh').encode() in argv
            birth=proc.joinpath('stat').read_text().split()[21]
            row=dict(pid=pid,birth=birth,epoch=time.time(),all_1200_tasks_completed=True,original_start_epoch=start,
                original_restore_epoch=start+20100,original_hard_cap_epoch=start+21600)
            (P/'attribution-pause.json').write_text(json.dumps(row,indent=2)+'\n');paused=True
            os.kill(pid,signal.SIGSTOP)
            for _ in range(100):
                if proc.joinpath('stat').read_text().split()[2] in ('T','t'):break
                time.sleep(.01)
            assert proc.joinpath('stat').read_text().split()[2] in ('T','t')
            status('full_pairing_complete_controller_paused')
            break
        time.sleep(.2)
    if paused:
        def guard():
            assert (P/'lease.active').exists() and time.time()<start+17400
            assert state('fnx-r162-rescue.service')=='inactive' and state('fnx-r162-run.service')=='active'
            proc=Path('/proc',str(pid));st=proc.joinpath('stat').read_text().split()
            assert st[21]==birth and st[2] in ('T','t')
        # The child writes its final summary before exiting; no other request is issued until it exits.
        for _ in range(100):
            alive=[]
            for proc in Path('/proc').glob('[0-9]*'):
                try:argv=proc.joinpath('cmdline').read_bytes().split(b'\0');cwd=proc.joinpath('cwd').resolve()
                except (FileNotFoundError,PermissionError,ProcessLookupError):continue
                if cwd==P and any(Path(os.fsdecode(x)).name=='quality_eval.py' for x in argv if x):alive.append(int(proc.name))
            if not alive:break
            time.sleep(.1)
        assert not alive,'QUALITY_CLIENT_STILL_ACTIVE'
        guard()
        # If the parent had already started stopping its Engine, skip the repeat rather than race it.
        try:
            h=json.load(urllib.request.urlopen('http://127.0.0.1:19002/health',timeout=3))
            assert h['diagnostic_variant']=='combined'
            repeat=post('/__r157_score',dict(natural=7,mode='prefill'))
            (P/'attribution-combined-repeat.json').write_text(json.dumps(repeat,indent=2)+'\n')
        except (OSError,AssertionError):repeat=None
        guard();status('stopping_completed_candidate')
        container('stop_owned.py',150);guard()
        runtime=json.loads((P/'runtime-code-manifest.json').read_text())['smooth']
        assert all(hashlib.sha256((P/'smooth/fnx'/n).read_bytes()).hexdigest()==h for n,h in runtime.items())
        container('launch_attribution.py',20);status('loading_w8a8_only_counterfactual')
        ready=False
        for _ in range(260):
            guard()
            try:
                h=json.load(urllib.request.urlopen('http://127.0.0.1:19002/health',timeout=2))
                ready=h['status']=='ok' and h['worker_alive'] and h['diagnostic_variant']=='smooth'
            except (OSError,ValueError):pass
            if ready:break
            time.sleep(2)
        assert ready,'ATTRIBUTION_ENGINE_NOT_READY'
        audit=post('/__r157_audit',{})
        layers=audit['parent']['layers']+audit['worker']['layers']
        assert len(layers)==48 and sum(x['type']=='SmoothW8A8' for row in layers for x in row['projections'].values())==96
        assert not audit['worker'].get('heads')
        (P/'attribution-runtime-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
        guard();status('scoring_same_zh65536_with_original_head')
        control=post('/__r157_score',dict(natural=7,mode='prefill'))
        (P/'attribution-smooth-result.json').write_text(json.dumps(control,indent=2)+'\n')
        lookup=lambda name:next(json.loads(x) for x in (P/name).read_text().splitlines() if json.loads(x)['case']==7)
        base=lookup('rollback-natural.jsonl');combined=lookup('combined-natural.jsonl')
        assert control['variant']=='smooth' and combined['variant']=='combined' and base['variant']=='rollback'
        for row in (control,combined,base):
            assert row['ids_sha256']==base['ids_sha256'] and row['mode']=='prefill' and not row['legacy']
            assert row['scored']==199 and len(row['nll_tokens'])==199 and all(math.isfinite(x) for x in row['nll_tokens'])
        mean=lambda row:math.fsum(row['nll_tokens'])/199
        if repeat:
            assert repeat['variant']=='combined' and repeat['ids_sha256']==base['ids_sha256'] and len(repeat['nll_tokens'])==199
        result=dict(epoch=time.time(),case=7,name='zh/65536',input_sha256=base['ids_sha256'],scored=199,
            baseline_nll=mean(base),w8a8_only_nll=mean(control),combined_nll=mean(combined),
            w8a8_projection_delta=mean(control)-mean(base),head_incremental_delta=mean(combined)-mean(control),
            combined_delta=mean(combined)-mean(base),
            repeated_combined_max_abs_token_delta=max(abs(a-b) for a,b in zip(repeat['nll_tokens'],combined['nll_tokens'])) if repeat else None,
            posthoc_attribution_only=True,original_quality_failure_not_reclassified=True,original_window_unchanged=True)
        (P/'attribution-complete.json').write_text(json.dumps(result,indent=2)+'\n')
        status('same_slice_attribution_complete',result=result)
        container('stop_owned.py',150)
finally:
    if paused:subprocess.run(['/usr/bin/python3',str(P/'recover_after_attribution.py')],check=True)
