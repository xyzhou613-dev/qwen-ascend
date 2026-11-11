"""r142: make the first-prefill-block size a parameter (default keeps 512).

Adds FNX_PIPELINE_FIRST_BLOCK to fnx/pipeline.py and p275.sh. Idempotent;
refuses to write when the expected anchors are missing.
"""
import sys
from pathlib import Path

PIPE = Path('/root/ascend-dev/fnx/fnx/pipeline.py')
P275 = Path('/root/ascend-dev/p275.sh')

ANCHOR_POLICY = ("        self.short_prefill_tokens = int(os.environ.get('FNX_PIPELINE_SHORT_TOKENS', '16384'))\n"
                 "        if not 0 <= self.short_prefill_chunk <= self.maximum or self.short_prefill_tokens < 1:\n")
ANCHOR_ADD = ("        if not 0 <= self.short_prefill_chunk <= self.maximum or self.short_prefill_tokens < 1:\n"
              "            raise ValueError('invalid short prefill chunk policy')\n")
NEW_ADD = ("        if not 0 <= self.short_prefill_chunk <= self.maximum or self.short_prefill_tokens < 1:\n"
           "            raise ValueError('invalid short prefill chunk policy')\n"
           "        # r142: first-block size while the short (dense-attention) regime is active.\n"
           "        # Every chunk re-reads all dense and expert weights, so 4x512 pays that\n"
           "        # fixed cost four times; one 2048 block covers the same tokens.\n"
           "        self.first_block = 2048 if os.environ.get('FNX_PIPELINE_FIRST_BLOCK', '0') == '1' else 512\n")
ANCHOR_USE = "            if self.stage0._host_pos<2048 and size>512:size=min(512,remaining)\n"
NEW_USE = "            if self.stage0._host_pos<2048 and size>self.first_block:size=min(self.first_block,remaining)\n"

ANCHOR_P275_DEF = "FNX_PIPELINE_SHORT_CHUNK=${FNX_PIPELINE_SHORT_CHUNK:-2048}\n"
NEW_P275_DEF = ANCHOR_P275_DEF + "FNX_PIPELINE_FIRST_BLOCK=${FNX_PIPELINE_FIRST_BLOCK:-0}\n"
ANCHOR_P275_USE = "FNX_PIPELINE_SHORT_CHUNK=$FNX_PIPELINE_SHORT_CHUNK FNX_PIPELINE_SHORT_TOKENS=16384"
NEW_P275_USE = "FNX_PIPELINE_SHORT_CHUNK=$FNX_PIPELINE_SHORT_CHUNK FNX_PIPELINE_FIRST_BLOCK=$FNX_PIPELINE_FIRST_BLOCK FNX_PIPELINE_SHORT_TOKENS=16384"


def patch(path, pairs):
    s = path.read_text()
    if 'FNX_PIPELINE_FIRST_BLOCK' in s and all(old not in s for old, _ in pairs):
        print('ALREADY_PATCHED', path)
        return
    for old, new in pairs:
        assert s.count(old) == 1, (path, old[:40], s.count(old))
        s = s.replace(old, new, 1)
    path.write_text(s)
    print('PATCHED', path)


def check():
    p = PIPE.read_text()
    assert ANCHOR_POLICY in p and ANCHOR_USE in p, 'pipeline anchors missing'
    q = P275.read_text()
    assert ANCHOR_P275_DEF in q and ANCHOR_P275_USE in q, 'p275 anchors missing'
    print('ANCHORS_OK')
    return True


if __name__ == '__main__':
    if '--check' in sys.argv:
        check()
        raise SystemExit(0)
    check()
    patch(PIPE, [(ANCHOR_ADD, NEW_ADD), (ANCHOR_USE, NEW_USE)])
    patch(P275, [(ANCHOR_P275_DEF, NEW_P275_DEF), (ANCHOR_P275_USE, NEW_P275_USE)])
    compile(PIPE.read_text(), str(PIPE), 'exec')
    print('PY_SYNTAX_OK')
