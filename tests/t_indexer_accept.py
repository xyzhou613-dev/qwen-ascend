"""QSA selection versus a literal per-query reference at budget boundaries."""
import math
import torch
from fnx import engine as E


def main():
    torch.manual_seed(49)
    dev = 'npu:1'
    P = 8192
    layer = E.FnLayer.__new__(E.FnLayer)
    layer.dev = dev
    layer.kv = type('KV', (), {'P': P})()
    layer.inv = E.inv_freq(dev)
    layer.index_w = E.Raw(torch.randn(640, E.H, dtype=torch.bfloat16, device=dev) * .01)
    layer.index_qn = torch.randn(128, dtype=torch.bfloat16, device=dev) * .1
    layer.index_kn = torch.randn(128, dtype=torch.bfloat16, device=dev) * .1
    history = torch.randn(P, 128, dtype=torch.bfloat16, device=dev)
    layer.index_k = history.clone()
    h = torch.randn(1, 1, E.H, dtype=torch.bfloat16, device=dev)
    def reference(pos):
        qk = E.lin(h, layer.index_w).reshape(640)
        q = E.rms_norm(qk[:512].reshape(4,128), layer.index_qn)
        cos, sin = E.rope_cs(torch.tensor([pos], device=dev), layer.inv)
        q = E.apply_rope(q, cos, sin)
        n = (pos+1)//4
        starts = torch.arange(n, device=dev)*4
        keys = layer.index_k[:n*4].view(n,4,128).float().mean(1).to(torch.bfloat16)
        keys = E.rms_norm(keys, layer.index_kn)
        cos, sin = E.rope_cs(starts, layer.inv)
        keys = E.apply_rope(keys, cos, sin)
        score = (q.float() @ keys.float().T).relu().sum(0)/math.sqrt(128)
        blocks = score.topk(min(512,n)).indices
        chosen = (blocks[:,None]*4+torch.arange(4,device=dev)).flatten()
        chosen = torch.cat((chosen,torch.arange(n*4,pos+1,device=dev)))
        mask = torch.zeros(P,dtype=torch.bool,device=dev)
        mask[chosen] = True
        return mask
    with torch.npu.device(dev):
        for pos in (0,3,2047,2048,2051,4095,8191):
            p = torch.tensor([pos],device=dev,dtype=torch.int64)
            mask = layer._indexer_mask(h,p)[0]
            ref = reference(pos)
            assert torch.equal(mask,ref), (pos,int((mask!=ref).sum()))
            print('MASK_PASS',pos,int(mask.sum()),flush=True)
        # Device positions must remain dynamic when replayed.
        p.fill_(2047)
        out = torch.empty(1,P,dtype=torch.bool,device=dev)
        stream = torch.npu.Stream()
        with torch.npu.stream(stream):
            out.copy_(layer._indexer_mask(h,p))
            torch.npu.synchronize(dev)
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph,capture_error_mode='relaxed'):
                out.copy_(layer._indexer_mask(h,p))
        for pos in (2047,2051,4095):
            p.fill_(pos)
            graph.replay()
            torch.npu.synchronize(dev)
            assert torch.equal(out[0],reference(pos)), pos
            print('GRAPH_MASK_PASS',pos,flush=True)
    print('INDEXER_ACCEPT',flush=True)


if __name__ == '__main__':
    with torch.no_grad():
        main()
