# r150 ordered q/k normalization

Accepted and deployed after exactness and speed gates. Read report.md, ab-summary.json and production-verification.json. Five documents/995 scored targets use the actual prefill policy; fixture IDs and source hashes are frozen. Three fresh 8k inputs and one near-capacity 256k input per version, plus a nonstream cached repeat. Production separately reproduces exact answers and speed gains.

Implementation is specialized to BF16 single-token contiguous [1,10240], head dimension128, CANN9.1 on 910B4. Other inputs fall back. Existing norm_post builds the CANN lookup table; do not substitute a generic rsqrt. ELF is Linux aarch64 and paired with the distributed source. Experimental controllers are historical server-specific operational evidence; do not execute without site checks and maintenance authorization. Full process ENV, credentials, model weights and graph caches are excluded. SHA256SUMS covers distributed files.
