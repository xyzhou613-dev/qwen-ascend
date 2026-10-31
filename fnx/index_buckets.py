"""Static graphs that exclude causally invisible future index keys."""
import os
import torch
from . import engine as E

def capture_index_buckets(m):
    if os.environ.get('FNX_INDEX_BUCKETS','0')!='1':return
    if os.environ.get('FNX_CHUNK_POOL')!='shared':
        raise ValueError('index buckets require serialized shared graph pools')
    caps=sorted(set(min(m.P,x) for x in (16384,65536,131072,m.P)))
    m.index_decode_graphs={m.P:m.g0}
    m.index_chunk_graphs={m.P:m.chunk_graphs}
    full_graph=m.g0;full_chunks=m.chunk_graphs
    stream=m._gs
    try:
        for cap in reversed(caps[:-1]):
            for L in m.layers0:L.index_scan_cap=cap
            m.reset()
            with torch.npu.stream(stream):
                m.body0();torch.npu.synchronize()
                g=torch.npu.NPUGraph()
                with torch.npu.graph(g,stream=stream,pool=full_graph.pool(),capture_error_mode='relaxed'):
                    m.body0()
            torch.npu.synchronize()
            m.index_decode_graphs[cap]=g
            m.capture_chunk()
            m.index_chunk_graphs[cap]=m.chunk_graphs
            torch.npu.empty_cache()
            print('[index-bucket] captured',cap,flush=True)
    finally:
        for L in m.layers0:L.index_scan_cap=m.P
        m.g0=full_graph;m.chunk_graphs=full_chunks
        for name,value in full_chunks[max(full_chunks)].items():setattr(m,name,value)
        m.reset()

def decode_graph(m,end):
    graphs=getattr(m,'index_decode_graphs',None)
    return graphs[min(cap for cap in graphs if cap>=end)] if graphs else m.g0

def chunk_graph(m,end,size):
    graphs=getattr(m,'index_chunk_graphs',None)
    bundles=graphs[min(cap for cap in graphs if cap>=end)] if graphs else m.chunk_graphs
    return bundles.get(size)
