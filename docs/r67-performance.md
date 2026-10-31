# r67: Bound FP32 DeltaNet preparation workspace for larger prefill chunks

Date: 2026-11-01 (Asia/Shanghai). Two Ascend 910B4 devices, 24+24 layers,
native 262144 capacity, batch=1. Server-local HTTP, loaded model, fresh UUID prompts.

Three fresh 8064-token prompts, 128 generated tokens each:
decode TPOT median 53.57 -> 53.27 ms/token
(0.56% higher decode throughput).

Fresh 262016-token prompt, 128 output tokens:
TTFT 111.226 s; decode 54.74 ms/token
(18.268 tokens/s); E2E 118.178 s.
Distributed fact retrieval: 3/3. Exact-prompt repeat also passed with zero prefill.
200-token teacher-forced quality: {'mean_nll': 2.165267, 'ppl': 8.7169, 'top1_acc': 0.485, 'n': 200}.

Evidence and candidate environment: benchmarks/r67/.
Reproduce with scripts/r51_payloads.py, scripts/r51_measure.py and
scripts/r52_long_client.py (TEST_CTX=262144 LONG_OUTPUT=128 LONG_EXPLAIN=1).
These measurements establish a local improvement, not a hardware ceiling.

The bounded preparation enables the 16384-token chunk that previously failed
with NPU OOM. State remains FP32; four random/correlated comparisons at 8192
and 16384 tokens are bit-exact against the deployed 8192-token preparation.
The 2048-token window was selected from five captured microbenchmarks.
The full 16384 chunk has worse TTFT than 8192, so this is an optional memory
optimization, not a recommendation to increase the production chunk size.
FNX_DELTA_WINDOW=0 preserves the original implementation.
