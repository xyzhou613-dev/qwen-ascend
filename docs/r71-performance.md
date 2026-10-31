# r71: Use native indexed attention with a lossless latent KV embedding

Date: 2026-11-01 (Asia/Shanghai). Two Ascend 910B4 devices, 24+24 layers,
native 262144 capacity, batch=1. Server-local HTTP, loaded model, fresh UUID prompts.

Three fresh 8064-token prompts, 128 generated tokens each:
decode TPOT median 52.62 -> 55.61 ms/token
(-5.38% higher decode throughput).

Fresh 262016-token prompt, 128 output tokens:
TTFT 104.908 s; decode 57.16 ms/token
(17.495 tokens/s); E2E 112.167 s.
Distributed fact retrieval: 3/3. Exact-prompt repeat also passed with zero prefill.
200-token teacher-forced quality: {'mean_nll': 2.165267, 'ppl': 8.7169, 'top1_acc': 0.485, 'n': 200}.

Evidence and candidate environment: benchmarks/r71/.
Reproduce with scripts/r51_payloads.py, scripts/r51_measure.py and
scripts/r52_long_client.py (TEST_CTX=262144 LONG_OUTPUT=128 LONG_EXPLAIN=1).
These measurements establish a local improvement, not a hardware ceiling.

This optional variant trades slower decode for lower TTFT. Native indexed
attention requires KV-head count 1, a 512-dimension latent and power-of-two
query groups on this A2 implementation. Two independent KV groups use padded
16-head queries. The latent packs K_nonrope(192), V(256), zero(64); Q packs
Q_nonrope(192), zero(320), with original RoPE(64) supplied separately. The
returned latent columns 192:448 are the original value result. No model data
is quantized or removed. Native arithmetic differs at BF16 rounding level
(~0.3% relative in isolated comparisons). Short attention reconstructs the
original K/V exactly. Cache storage grows 12.5%, from 512 to 576 BF16 values
per KV head/position. This is opt-in through FNX_NATIVE_SPARSE=1.
