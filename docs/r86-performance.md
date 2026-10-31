# r86: Select index graphs by complete causal groups and refine capacities

2026-11-01, two Ascend 910B4 devices, loaded 256k model, server-local HTTP.
Fresh 262016-token input + 128 generated tokens; prefill 262012, fact retrieval 3/3.

TTFT: 103.956 -> 103.364 seconds.
Decode TPOT: 53.88 -> 54.17 ms
(18.46 tokens/s).
Three fresh 8k requests: median TTFT 5.910 s,
TPOT 52.81 ms.
200-token quality: {'mean_nll': 2.165267, 'ppl': 8.7169, 'top1_acc': 0.485, 'n': 200}. Original NLL gate remains 2.164792 +/-0.01.
The exact-prompt repeat passed with zero prefill and all three facts correct.

Only complete four-token groups participate in index scoring. The final
0..3 tokens use the existing causal tail. Graph selection therefore rounds
the exclusive query end down to four, rather than unnecessarily advancing
to the next capacity for up to three partial-group tokens. KV capacity,
query positions and tail attention remain unchanged. Added 32k, 96k and
192k score capacities keep aligned matrix dimensions. Twenty-four isolated
boundary comparisons preserve selected token IDs/masks exactly. Full model
quality and three-fact retrieval also pass. Capture remains serialized on
the same stream/pool; extra graphs increase startup time and memory.
