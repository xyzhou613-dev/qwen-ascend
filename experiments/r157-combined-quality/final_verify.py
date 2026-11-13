"""Read-only verification after the controller and guardian have finished."""
import hashlib
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

P = Path(__file__).resolve().parent
LIVE = Path('/root/ascend-dev/fnx/fnx')


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def main():
    before = json.loads((P/'production-before-hashes.json').read_text())
    after = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in LIVE.iterdir()
             if f.is_file() and f.suffix in ('.py', '.so')}
    assert before == after, 'production source or ELF changed'
    harness = json.loads((P/'quality-harness-manifest.json').read_text())['files']
    assert all(hashlib.sha256((P/name).read_bytes()).hexdigest() == r['sha256']
               for name, r in harness.items()), 'quality harness changed during pairing'
    runtime = json.loads((P/'runtime-code-manifest.json').read_text())
    for phase, files in runtime.items():
        assert all(hashlib.sha256((P/phase/'fnx'/name).read_bytes()).hexdigest() == digest
                   for name, digest in files.items()), ('private runtime changed during pairing', phase)
    start = Path('/root/ascend-dev/p275.sh').read_bytes()
    assert start == (P/'p275-before.sh').read_bytes(), 'formal startup script changed'
    health = json.load(urllib.request.urlopen('http://127.0.0.1:9002/health', timeout=5))
    assert health['status'] == 'ok' and health['worker_alive'] and health['service_revision'] == 'r152'
    assert health['ctx'] == 262144 and 'diagnostic_variant' not in health
    smoke_path=P/'production-http.json'
    smoke=json.loads(smoke_path.read_text())
    assert smoke_path.stat().st_mtime >= (P/'restored-health.json').stat().st_mtime
    assert smoke['answer_matches_baseline'] and smoke['stats']['tokens']==128 and smoke['stats']['n_prefill']>=8060
    identity = json.loads(Path('/root/ascend-dev/fnx-boot-restore/container-identity.json').read_text())
    cid = command('docker', 'inspect', '-f', '{{.Id}}', 'bole-model-Flash')
    assert cid == identity['container_id'], 'original container replaced'
    judge = json.loads((P/'judge-image.json').read_text())
    assert command('docker', 'image', 'inspect', '-f', '{{.Id}}', judge['tag']) == judge['image_id'], 'CPU judge image tag changed'
    engines, watchdogs, diagnostic = [], [], []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            args = proc.joinpath('cmdline').read_bytes().split(b'\0')
            cwd = os.readlink(proc/'cwd')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if b'fnx.server' in args or b'fnx.pipeline_worker' in args:
            engines.append(dict(pid=int(proc.name), cwd=cwd,
                                module='server' if b'fnx.server' in args else 'worker'))
        if args and Path(os.fsdecode(args[0])).name in ('bash', 'sh') and b'/root/ascend-dev/watchdog.sh' in args:
            watchdogs.append(int(proc.name))
        if (cwd.startswith(str(P)+'/') and (b'fnx.server' in args or b'fnx.pipeline_worker' in args)) or any(a.endswith(b'/profile_model.py') and a.startswith(str(P).encode()+b'/') for a in args):
            diagnostic.append(int(proc.name))
        if b'/models/fnx-diagnostics/r158-bf16-down-20261113/numeric_probe.py' in args:
            diagnostic.append(int(proc.name))
    assert len(engines) == 2 and {r['module'] for r in engines} == {'server', 'worker'}, engines
    assert all(r['cwd'] == '/root/ascend-dev/fnx' for r in engines), engines
    assert len(watchdogs) == 1 and not diagnostic, (watchdogs, diagnostic)
    assert not command('docker', 'ps', '-q', '--filter', 'label=fnx.diagnostic=r157'), 'temporary CPU judges still running'
    assert command('systemctl', 'is-active', 'model-switch.service') == 'active'
    assert command('systemctl', 'show', 'model-switch.service', '--value', '-p', 'RefuseManualStart') == 'no'
    assert 'r157' not in command('systemctl', 'show', 'model-switch.service', '--value', '-p', 'DropInPaths')
    assert not Path('/run/systemd/system/fnx-r157-restore-dispatch.service.d/r157-dispatch.conf').exists()
    for unit in ('fnx-r157-restore-dispatch.timer','fnx-r157-rescue.timer'):
        assert not (Path('/run/systemd/system')/(unit+'.d')/'r157-deadline.conf').exists()
    assert command('systemctl', 'is-enabled', 'fnx-boot-restore.service') == 'enabled'
    for name in ('lease.active', 'original-private.json'):
        assert not (P/name).exists(), ('experimental private state left behind', name)
    states = {}
    for unit in ('fnx-r157-run', 'fnx-r157-run-resume', 'fnx-r157-safety',
                 'fnx-r157-safety-resume', 'fnx-r157-rescue.timer', 'fnx-r157-rescue.service',
                 'fnx-r157-precap.timer', 'fnx-r157-precap.service',
                 'fnx-r157-restore-dispatch.timer', 'fnx-r157-restore-dispatch.service',
                 'fnx-r157-canonical-check', 'fnx-r158-build', 'fnx-r158-numeric', 'fnx-r158-numeric-retry'):
        state = command('systemctl', 'show', unit, '--value', '-p', 'ActiveState')
        states[unit] = state
        assert state not in ('active', 'activating', 'deactivating'), (unit, state)
    result = dict(verified_epoch=time.time(), production_files_unchanged=len(after),
                  p275_sha256=hashlib.sha256(start).hexdigest(), original_container_preserved=True,
                  health=health, engines=engines, watchdog_pids=watchdogs,
                  manager_active=True, experimental_dropin_absent=True,
                  quality_harness_files_unchanged=len(harness), private_runtimes_unchanged=True,
                  cpu_judge_image_unchanged=True,
                  formal_fresh_8k_http_smoke_passed=True,
                  experimental_processes_absent=True, private_env_removed=True,
                  experiment_unit_states=states, normal_boot_recovery_enabled=True,
                  boot_changed=(P/'original-boot-id').read_text().strip() != Path('/proc/sys/kernel/random/boot_id').read_text().strip())
    window_start = int((P/'window-start.txt').read_text())
    result['window_start_epoch'] = window_start
    result['canonical_restore_elapsed_s'] = (P/'restored-health.json').stat().st_mtime-window_start
    result['restored_within_six_hour_cap'] = result['canonical_restore_elapsed_s'] <= 21600
    (P/'final-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print('FINAL_RESTORE_VERIFIED', len(after), 'unchanged model files')


if __name__ == '__main__':
    main()
