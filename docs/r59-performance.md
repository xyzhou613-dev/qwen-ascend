# r59: Use selected-expert grouped matmul for decode down projection

Date: 2026-11-01 (Asia/Shanghai). Two Ascend 910B4 devices, 24+24 layers,
native 262144 capacity, batch=1. Server-local HTTP, loaded model, fresh UUID prompts.

Three fresh 8064-token prompts, 128 generated tokens each:
decode TPOT median 54.80 -> 54.19 ms/token
(1.13% higher decode throughput).

Fresh 262016-token prompt, 128 output tokens:
TTFT 116.823 s; decode 54.21 ms/token
(18.447 tokens/s); E2E 123.708 s.
Distributed fact retrieval: 3/3. Exact-prompt repeat also passed with zero prefill.
200-token teacher-forced quality: {'mean_nll': 2.165267, 'ppl': 8.7169, 'top1_acc': 0.485, 'n': 200}.

Evidence and candidate environment: benchmarks/r59/.
Reproduce with scripts/r51_payloads.py, scripts/r51_measure.py and
scripts/r52_long_client.py (TEST_CTX=262144 LONG_OUTPUT=128 LONG_EXPLAIN=1).
These measurements establish a local improvement, not a hardware ceiling.
