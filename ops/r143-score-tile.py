"""r143: column-tiled index scoring (bitwise: same per-column dot products).

Scores query tiles against the key cache in cache-sized column slices so the
[4*n, groups] FP32 intermediate does not have to be written to HBM whole.

    python3 r143-score-tile.py /root/ascend-dev --check
    python3 r143-score-tile.py /root/ascend-dev
"""
import sys
from pathlib import Path

ANCHOR_FLAG = '_DN_RMS = os.environ.get("FNX_DN_RMS", "1") == "1"'
NEW_FLAG = (ANCHOR_FLAG + "\n"
            '# r143: indexer score column tile (0 = whole row). Bitwise: columns are\n'
            '# independent in both the matmul and the relu/sum epilogue.\n'
            '_SCORE_COLS = int(os.environ.get("FNX_INDEX_SCORE_TILE", "0"))')

ANCHOR_LOOP = """            scores = torch.matmul(q[begin:end].float().reshape(-1, 128), keys)
            reduced = None
            if fuse_scores and end - begin >= 128:
                reduced = native_score(scores, p)
            if reduced is None:
                reduced = scores.view(end - begin, 4, groups).relu().sum(1)
            scores = reduced / math.sqrt(128)
"""
NEW_LOOP = """            qt = q[begin:end].float().reshape(-1, 128)
            n = end - begin
            reduced = None
            if (fuse_scores and n >= 128 and _SCORE_COLS
                    and groups > _SCORE_COLS and groups % _SCORE_COLS == 0):
                reduced = torch.empty((n, groups), dtype=torch.float32, device=qt.device)
                for c0 in range(0, groups, _SCORE_COLS):
                    c1 = c0 + _SCORE_COLS
                    piece = native_score(torch.matmul(qt, keys[:, c0:c1]), p)
                    if piece is None:
                        reduced = None
                        break
                    reduced[:, c0:c1] = piece
            if reduced is None:
                scores = torch.matmul(qt, keys)
                if fuse_scores and n >= 128:
                    reduced = native_score(scores, p)
                if reduced is None:
                    reduced = scores.view(n, 4, groups).relu().sum(1)
            scores = reduced / math.sqrt(128)
"""

ANCHOR_P275_DEF = "FNX_PIPELINE_SHORT_CHUNK=${FNX_PIPELINE_SHORT_CHUNK:-2048}\n"
NEW_P275_DEF = ANCHOR_P275_DEF + "FNX_INDEX_SCORE_TILE=${FNX_INDEX_SCORE_TILE:-0}\n"
ANCHOR_P275_USE = "FNX_INDEX_TILE=512"
NEW_P275_USE = "FNX_INDEX_TILE=512 FNX_INDEX_SCORE_TILE=$FNX_INDEX_SCORE_TILE"


def patch(path, pairs):
    s = path.read_text()
    if 'FNX_INDEX_SCORE_TILE' in s:
        print('ALREADY_PATCHED', path)
        return
    for old, new in pairs:
        assert s.count(old) == 1, (str(path), old[:48], s.count(old))
        s = s.replace(old, new, 1)
    path.write_text(s)
    print('PATCHED', path)


def main():
    root = Path(sys.argv[1])
    check_only = '--check' in sys.argv[2:]
    eng = root / 'fnx/fnx/engine.py'
    p275 = root / 'p275.sh'
    assert ANCHOR_LOOP in eng.read_text(), 'engine loop anchor missing'
    assert ANCHOR_FLAG in eng.read_text(), 'engine flag anchor missing'
    q = p275.read_text()
    assert ANCHOR_P275_DEF in q and ANCHOR_P275_USE in q, 'p275 anchor missing'
    print('ANCHORS_OK')
    if check_only:
        return
    patch(eng, [(ANCHOR_FLAG, NEW_FLAG), (ANCHOR_LOOP, NEW_LOOP)])
    patch(p275, [(ANCHOR_P275_DEF, NEW_P275_DEF), (ANCHOR_P275_USE, NEW_P275_USE)])
    compile(eng.read_text(), str(eng), 'exec')
    print('PY_SYNTAX_OK')


if __name__ == '__main__':
    main()
