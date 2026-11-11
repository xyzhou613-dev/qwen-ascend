# Exact MoE Cartesian output epilogue

Build kernels/moe_epilogue/build.sh with the validated CANN toolchain. Install engine.py, server.py, moe_epilogue.py and libfnx_moe_epilogue.so together; opt in with FNX_MOE_EPILOGUE=1. Framework default is off.

Requires the existing BF16 Cartesian down path (FNX_DECODE_DN_GMM=0), FNX_MOE_REDUCE=1 and the r140 deferred shared gate. Other branches retain their reference paths. Matrix multiplication, weights, routing and prefill are unchanged.

The kernel reads the 10 diagonal expert blocks directly, preserving BF16 product rounding, ordered FP32 accumulation, BF16 reduced output, BF16 shared sigmoid/product, and final BF16 addition. Intermediate global diagonal, weight conversion and reduced-output tensors are eliminated.

128 dynamic component inputs and real L00 T=1/128/512/2048 (8 dynamic inputs each) passed bitwise output/state/history checks. Graph component timings are not model speedups. Full request evidence is added only after acceptance.

With an installed/built runtime, run from repository root using PYTHONPATH="$PWD" python benchmarks/r141-moe-epilogue/probe.py. All native wrappers acquire the current NPU stream for each call; input preparation must be submitted before raw CANN launches. Use isolated NPU timing for final calibration.

Formal production acceptance: 8k median 34.75ms, 256k 35.99ms; see summary.json and docs/r141-moe-epilogue.md.
