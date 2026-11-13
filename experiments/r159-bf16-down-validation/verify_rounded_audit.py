import json,sys
from pathlib import Path
P=Path(__file__).resolve().parent;phase=sys.argv[1]
if phase=='combined':
    a=json.loads((P/'combined-runtime-audit.json').read_text())
    for side in ('parent','worker'):
        assert a[side]['rounded_library']==str(P/'combined/fnx/libfnx_w4_bf16w.so')
        assert a[side]['rounded_tile']==(P/'selected-tile.txt').read_text().strip()
    print('BF16_ROUNDED_LIBRARY_VERIFIED_IN_BOTH_PROCESSES',flush=True)
