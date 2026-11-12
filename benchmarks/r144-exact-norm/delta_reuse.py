import torch
import torch.nn.functional as F
NVH=48
HKD=HVD=128
_CACHE={}
def constants(device):
    key=str(device)
    if key not in _CACHE:
        upper=torch.ones(64,64,dtype=torch.bool,device=device).triu(1)
        diag=torch.arange(8,device=device)*9
        eye=torch.eye(8,device=device)
        bid=torch.arange(64,device=device)//8
        mask=~(bid[:,None]>bid[None,:])
        _CACHE[key]=(upper,diag,eye,mask)
    return _CACHE[key]

def delta_chunk_batched_fp32(q, k, v, log_g, beta, state):
    """Prepare independent 64-token systems together; keep state recurrence fp32.

    The systems depend on keys/decay, not on the incoming recurrent state.
    Only the state update and output need a sequential loop over blocks.
    """
    T = q.shape[0]
    blocks = (T + 63) // 64
    length = blocks * 64
    def rows(x, dim):
        x = F.pad(x, (0, 0, 0, 0, 0, length - T))
        return x.view(blocks, 64, NVH, dim).permute(0, 2, 1, 3).reshape(-1, 64, dim).float()
    qa, ka, va = (rows(x, d) for x, d in ((q, HKD), (k, HKD), (v, HVD)))
    ba = F.pad(beta.reshape(T, NVH), (0, 0, 0, length - T))
    ba = ba.view(blocks, 64, NVH).transpose(1, 2).reshape(-1, 64).float()
    ga = F.pad(log_g, (0, 0, 0, length - T))
    cumul = ga.view(blocks, 64, NVH).transpose(1, 2).reshape(-1, 64).cumsum(-1)
    upper, diag_index, inv8, block_mask = constants(q.device)
    decay = (cumul.unsqueeze(-1) - cumul.unsqueeze(-2)).masked_fill(upper, float('-inf')).exp()
    kb, vb = ka * ba.unsqueeze(-1), va * ba.unsqueeze(-1)
    lower = (torch.bmm(kb, ka.transpose(1, 2)) * decay).tril(-1)
    cexp = cumul.exp()
    rhs = torch.cat((vb, kb * cexp.unsqueeze(-1)), -1)
    heads = blocks * NVH
    diagonal = lower.view(heads, 8, 8, 8, 8).permute(0, 1, 3, 2, 4).reshape(heads, 64, 8, 8)
    diagonal = diagonal.index_select(1, diag_index).reshape(heads * 8, 8, 8)
    inv = inv8.expand(heads * 8, 8, 8)
    power = -diagonal
    for stage in range(3):
        inv = inv + torch.bmm(power, inv)
        if stage < 2:
            power = torch.bmm(power, power)
    solved = torch.bmm(inv, rhs.reshape(heads * 8, 8, HVD + HKD)).reshape(heads, 64, HVD + HKD)
    transformed = torch.bmm(inv, lower.reshape(heads * 8, 8, 64)).reshape(heads, 64, 64)
    # Cached inter-block mask.
    power = -transformed.masked_fill(block_mask, 0)
    for stage in range(3):
        solved = solved + torch.bmm(power, solved)
        if stage < 2:
            power = torch.bmm(power, power)
    u = solved[..., :HVD].reshape(blocks, NVH, 64, HVD)
    w = solved[..., HVD:].reshape(blocks, NVH, 64, HKD)
    intra = (torch.bmm(qa, ka.transpose(1, 2)) * decay).reshape(blocks, NVH, 64, 64)
    qe = (qa * cexp.unsqueeze(-1)).reshape(blocks, NVH, 64, HKD)
    end_keys = (ka * (cumul[:, -1:] - cumul).exp().unsqueeze(-1)).reshape(blocks, NVH, 64, HKD)
    end_decay = cumul[:, -1:].exp().reshape(blocks, NVH, 1, 1)
    current = state[0]
    outputs = []
    for block in range(blocks):
        updates = u[block] - torch.bmm(w[block], current)
        out = torch.bmm(qe[block], current) + torch.bmm(intra[block], updates)
        current = current * end_decay[block] + torch.bmm(end_keys[block].transpose(1, 2), updates)
        outputs.append(out.transpose(0, 1))
    return torch.cat(outputs, 0)[:T].to(torch.bfloat16), current.unsqueeze(0)
