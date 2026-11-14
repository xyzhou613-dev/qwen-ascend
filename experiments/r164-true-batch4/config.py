import json,os
from pathlib import Path
P=Path(__file__).resolve().parent
def environment(device='0,1',batch=4):
    env=os.environ.copy()
    env.update(json.loads((P/'selected-config.json').read_text()))
    env.update(ASCEND_RT_VISIBLE_DEVICES=str(device),OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',
        PYTHONPATH=str(P/'runtime')+':'+str(P),FNX_BATCH=str(batch),FNX_CTX='8192',
        FNX_PIPELINE='0',FNX_CHUNK='512',FNX_CHUNK_GRAPH='0',FNX_CHUNK_SIZES='',
        FNX_GRAPH_G1='layer',FNX_HEAD_DEV='1',FNX_MOE_GROUP='1',FNX_W4_SCALE_CACHE='1',
        FNX_MOE_GROUP_SPLIT='0',FNX_NO_NGRAM='1',FNX_NATIVE_SPARSE='0',
        FNX_NATIVE_DUAL='0',FNX_DUAL_WRITE='0',FNX_KV_POSITION_MAJOR='0',
        FNX_POST_NATIVE='0',FNX_NORM_POST='0',FNX_NORM_ORDERED='0',
        FNX_INJECT_NATIVE='0',FNX_MIX_REDUCE='0',FNX_MOE_REDUCE='0',
        FNX_MOE_EPILOGUE='0',FNX_CONV_NATIVE='0',FNX_DN_NATIVE='0',
        FNX_INDEX_POST='0',FNX_INDEX_EXPAND='0',FNX_INDEX_SCORE='0',
        FNX_DN_DECODE_BMM='0',FNX_EXPERIMENT_BATCH_ROWS='1')
    return env
