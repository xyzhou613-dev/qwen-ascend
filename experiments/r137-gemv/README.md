# Rejected scalar-row BF16 vector GEMV

Not enabled in inference. It was slower than CANN MatMul with both repeated weights and a 512 MiB rotating weight pool. Reducing host graph-launch noise exposed the slowdown. The first single-op-per-replay timing was invalid for comparing microsecond kernels and was excluded.

FP32 reductions also changed some BF16 output bits, so the candidate did not meet an exactness criterion. The useful result is the negative experiment and the cache/dispatch controls, not an inference speedup.

Build gemv.cpp with the same AscendC toolchain as kernels/inject/build.sh, producing libgemv.so beside probe.py. Set COLD_MIB=512 for the rotating pool. The last graph output is checked dynamically, and the timing averages 128 operations per graph. This is a standalone projection probe, not a measured full-model result.

For the tiny 48x2560 case, the 128-call graph only visits 128 weights from the allocated pool; do not describe that case as a full 512 MiB cold working set. The larger projection cases visit the full rotating pool.
