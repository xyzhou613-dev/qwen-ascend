# r137 exact decode residual injection

Opt-in: build kernels/inject/build.sh in the target CANN environment, then use FNX_INJECT_NATIVE=1. Default is off; unsupported shapes/dtypes/layouts retain the existing path.

The kernel preserves BF16 product rounding before adding the residual. It must not be replaced with one floating-point FMA. The public wrapper only accepts batch=1, hyper=[1,1,10240], 2560 values and four gates. Large prefill uses the old path: the native microkernel is slower at batch=128.

Validation: 32 dynamic micro inputs, real L00 at T=1/128/512/2048 (8 inputs each), exact output/state/history, and three identical 8064-input/128-output requests with byte-identical output. Candidate median decode TPOT 42.78 ms versus baseline 43.19 ms. 200-token NLL=2.165267. Long-context/deployment acceptance is recorded separately in docs/r137-decode-inject.md after completion.

Build and run the benchmark in the existing torch_npu environment; the graph captures 128 repetitions to avoid timing host replay dispatch as kernel cost. Full HTTP request measurements supersede extrapolation from this microbenchmark.

Production acceptance: 8k median TPOT 42.50 ms, 256k TPOT 43.87 ms / HTTP TTFT 73.703 s, NLL 2.165267, exact 8k outputs and 3/3 long-context recall. The switch manager is active and temporary start restrictions were removed.
