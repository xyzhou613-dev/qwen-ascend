"""On a new boot, undo an unaccepted source update before normal restoration."""
import json,subprocess,time
from pathlib import Path
P=Path(__file__).resolve().parent
if not (P/'lease.active').exists() or (P/'accepted.json').exists():raise SystemExit(0)
w=json.loads((P/'window.json').read_text());boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
if boot==w['boot_id']:raise SystemExit(0)
for proc in Path('/proc').glob('[0-9]*'):
    try:args=proc.joinpath('cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,PermissionError,ProcessLookupError):continue
    assert b'fnx.server' not in args and b'fnx.pipeline_worker' not in args,'REFUSED_ACTIVE_ENGINE_AT_BOOT'
subprocess.run(['python3',str(P/'apply_release.py'),'rollback'],check=True)
(P/'boot-rollback-complete.json').write_text(json.dumps(dict(epoch=time.time(),previous_boot=w['boot_id'],current_boot=boot,source_restored_before_model_start=True),indent=2))
(P/'original-private.json').unlink(missing_ok=True);(P/'lease.active').unlink()
