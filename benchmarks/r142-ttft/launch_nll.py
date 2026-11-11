import json
import os
import subprocess
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
d = json.loads((P / 'original-private.json').read_text())
env = d['env'].copy()
env['PYTHONPATH'] = str(P) + os.pathsep + env.get('PYTHONPATH', '')
env['FNX_PIPELINE_FIRST_BLOCK'] = sys.argv[1]
sys.exit(subprocess.call([d['cmd'][0], str(P / 'prefill_nll.py'), sys.argv[1]], cwd=P, env=env))
