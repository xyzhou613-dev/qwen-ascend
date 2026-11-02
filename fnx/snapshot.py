"""Reconstruct redundant native MLA caches from the exact original BF16 K/V.

Snapshots may omit packed tensors; live dual caches and their addresses remain
unchanged for captured graphs. Restoration uses copies only, without rounding.
"""

def rebuild_packed(store, valid_len, tile=8192):
    if not 0 <= valid_len <= store.P:
        raise ValueError("snapshot length outside KV capacity")
    if not getattr(store, "native_dual", False):
        raise ValueError("packed reconstruction requires native dual KV")
    for layer in range(store.n):
        key, value = store.k[layer], store.v[layer]
        latent, rope = store.k[layer+store.n], store.v[layer+store.n]
        for begin in range(0, valid_len, tile):
            end = min(begin+tile, valid_len)
            latent[:, :, begin:end, :192].copy_(key[:, :, begin:end, 64:])
            latent[:, :, begin:end, 192:448].copy_(value[:, :, begin:end])
            latent[:, :, begin:end, 448:].zero_()
            rope[:, :, begin:end].copy_(key[:, :, begin:end, :64])
