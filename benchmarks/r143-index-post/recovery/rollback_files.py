from pathlib import Path
import json,shutil
P=Path(__file__).resolve().parent
for row in json.loads((P/'deployment-backup/manifest.json').read_text()):
 target=Path(row['target'])
 assert target.parent==Path('/root/ascend-dev/fnx/fnx') or target==Path('/root/ascend-dev/p275.sh')
 if row['existed']:shutil.copy2(row['backup'],target)
 elif target.exists():target.unlink()
print('EXACT_PRE_R143_FILES_RESTORED',flush=True)
