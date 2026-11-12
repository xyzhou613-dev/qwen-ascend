# r143 exact FP32 index score postprocessing

The runtime flag FNX_INDEX_POST=1 enables index-score scaling and causal masking. It uses the original four-head sum when that native path was already selected; otherwise it consumes the unchanged torch reduction. Unsupported shapes retain the original path. The r142 first-block and all other prefill policies are unchanged.

Build kernels/index_post/build.sh under the validated CANN toolchain. Install engine.py, server.py, index_post.py and libfnx_index_post.so together. The framework default is off. Production uses mode1 Div; mode0 Muls is a rejected calibration alternative and is not used by the engine.

The calibration compares FP32 score bits and TopK indices against the old production chain. It includes 36 shape/position cases and 2,097,152 wide-exponent nonnegative FP32 values. Mode1 is exact; mode0 differs and is rejected. Real full-attention L03 at context capacity65536/position61440, T=1/128/512/2048, eight inputs each, passed output/KV/raw-index/cached-index bitwise checks.

The component probe can use FNX_SCORE_SO pointing to the installed reference libfnx_score.so and FNX_INDEX_POST_SO pointing to the newly built candidate library. Full-layer tests require the complete production environment and real weights, and must run under the guarded exclusive harness. Do not start another complete Engine beside the deployed model.

Natural prefill NLL follows r142: prefill5800 tokens using the production policy, then score199 next-token targets in the final200-token tail of the same6000-token natural document. The candidate reuses its already-loaded model through a diagnostic-only route; that route is not installed in production. Formal model verification uses source/binary hashes, byte-equal8k answers, and fresh256k retrieval/cache tests. /fnx/loss remains a decode-only check.

A deployed gather check using its production wrapper and synchronized prepared inputs passed18/18 bitwise reference comparisons. The older cached-stream/zeroing probes are insufficient to establish a production binary defect. Binary SHA differences alone also do not establish semantic differences. The gather binary is unchanged.
