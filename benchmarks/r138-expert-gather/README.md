# r138 selected-expert gather fusion

Build kernels/expert_gather/build.sh with the target CANN environment and enable FNX_EXPERT_GATHER=1 after installing the shared library. Default off; unsupported shapes/layouts and grouped prefill retain the existing path. Baseline runtime is r137.

The kernel copies expert-major W4 packed bytes and cached BF16 scale bytes directly into the K-major layout consumed by the unchanged CANN up projection. It preserves the original top-10 order and does not unpack weights or recompute scales. Current wrapper supports only E512/K2560/N1280/G40/top10.

32 dynamic ids/inputs (including repeated and edge ids) were checked, plus real L00 at T1/128/512/2048 with eight dynamic inputs each. Output, FP32 state and BF16 history are exact. Complete up-path synthetic timing is 98.75 -> 52.61 us; whole-layer decode about .781 -> .693 ms. HTTP production results are documented separately after acceptance.

The six-way comparison includes the deployed direct-copy implementation, the CANN baseline, and unselected UB-gather/contiguous-write prototypes. To rerun probe.py, build experiments/r138-layout/gather.cpp to libgather.so beside this benchmark, and build the production library to fnx/libfnx_expert_gather.so (or set FNX_EXPERT_GATHER_SO). Graphs contain 16 repeated operations. Standalone direct gather near 18.55 us is still close to the host replay floor and is an upper-bound timing reference. Full up and full-model tests decide acceptance. layer_probe.py requires the runtime environment and real L00 artifacts.

Production: 8k median TPOT 39.16 ms versus r137 42.60 ms; 256k TPOT 40.37 ms / HTTP TTFT 73.568 s. Outputs, NLL 2.165267, recall and cache checks passed. Health revision r138 with both flags true; manager active and temporary restrictions removed.
