# r140 exact decode postprocessing

Build kernels/post_ops/build.sh, install fnx/post_ops.py and libfnx_post.so with the corresponding engine source, then enable FNX_POST_NATIVE=1. Default off. Wrappers restrict shapes/dtypes/layouts to the verified single-token cases; other inputs retain the old path. No RMS reduction or weight values are changed.

Functions: HC BF16 division/Silu/injection gate; sigmoid plus four-branch rounded weighted mean; post-RMS FP32 multiply/cast (retaining the BF16 normalization boundary for DeltaNet); and shared-expert sigmoid/multiply/add merge with BF16 product rounding. Activation always uses mode0 (Div); reciprocal/reassociation alternatives in the calibration are not used.

All 65280 finite BF16 bit patterns were covered for HC activation/gate and for sigmoid-gate roles; 32 dynamic tests passed for activation/mix/MoE merge, and 64 for each post-multiply variant. Real L00 T1/128/512/2048 with eight dynamic inputs each passed exact output/state/history checks. Three identical 8k HTTP outputs matched and NLL=2.165267. Full production evidence is in docs/r140-post.md after acceptance.

Micro timings are not extrapolated to model speed. Standalone normalization Mul+Cast was nearly flat (2.38 ->2.21us), while activation, mix and rounded DeltaNet gating reduced calls. All graph microbenchmarks amortize launch overhead over 256/512 operations.

layer_probe.py runs the component checks first and needs the same runtime profile and real L00 artifact. Build/install libfnx_reduce.so for its reference implementation. Optional FNX_POST_SO/FNX_REDUCE_SO override library locations. Runtime integration releases pre-sigmoid logits immediately after their consumer to preserve large-prefill temporary lifetimes.

Formal acceptance: production8k median 36.88ms; production256 38.12ms. summary.json includes bracketing baselines, request metrics and answer SHA256. Final-build component and layer evidence replaces earlier exploratory timings. See docs/r140-post.md.
