# r78: Keep both native sparse and compact decode KV layouts coherent

Date: 2026-11-01 (Asia/Shanghai). Two Ascend 910B4 devices, 24+24 layers,
native 262144 capacity, batch=1. Server-local HTTP, loaded model, fresh UUID prompts.

Three fresh 8064-token prompts, 128 generated tokens each:
decode TPOT median 53.35 -> 52.61 ms/token
(1.41% higher decode throughput).

Fresh 262016-token prompt, 128 output tokens:
TTFT 105.061 s; decode 54.01 ms/token
(18.515 tokens/s); E2E 111.920 s.
Distributed fact retrieval: 3/3. Exact-prompt repeat also passed with zero prefill.
200-token teacher-forced quality: {'mean_nll': 2.165267, 'ppl': 8.7169, 'top1_acc': 0.485, 'n': 200}.

Evidence and candidate environment: benchmarks/r78/.
Reproduce with scripts/r51_payloads.py, scripts/r51_measure.py and
scripts/r52_long_client.py (TEST_CTX=262144 LONG_OUTPUT=128 LONG_EXPLAIN=1).
These measurements establish a local improvement, not a hardware ceiling.

FNX_NATIVE_DUAL=1 retains position-major original K/V and the packed latent
cache. Every write updates both; flattened (head,position) indices combine
the two packed heads into one update per cache. Both representations live
in the existing KV lists, so reset, save, restore and conversation reuse
include both automatically. Decode reads original K/V directly, with no
latent reconstruction. All values remain BF16; recurrent state stays FP32.
The additional original cache uses 3 GiB per device at 256k, plus the same
amount in a full GPU conversation snapshot. This favors decode speed;
r75 without dual layouts retains slightly lower TTFT and lower memory use.
