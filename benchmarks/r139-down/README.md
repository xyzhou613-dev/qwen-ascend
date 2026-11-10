# r139 decode down projection

Baseline is r138. The selected path reuses the existing Cartesian down projection and diagonal extraction, now feeding it directly packed K-major weights/scales through the generalized gather kernel. Set FNX_EXPERT_GATHER=1 and FNX_DECODE_DN_GMM=0 after installing the updated library. Restore FNX_DECODE_DN_GMM=1 to use the r138 grouped down path.

The library now accepts gate/up E512/K2560/N1280/G40 and down E512/K640/N2560/G10. Only 2D activations with one or TOPK rows use direct gather; grouped prefill is unchanged.

32 dynamic synthetic inputs and real L00 T1/128/512/2048 with eight inputs each passed exact output/state/history checks. Three identical 8k requests gave byte-identical output and NLL=2.165267. Micro down path 102.2 -> 91.6 us; L00 decode about .699 -> .650 ms. Request acceptance is recorded in docs/r139-down.md.

Build kernels/expert_gather/build.sh, then run diag_probe.py in the existing torch_npu environment. Each graph contains 16 repetitions to amortize host replay overhead. The literal intermediate output is 10x10 expert blocks instead of 10 diagonal blocks; extra logical work does not establish the physical padded Cube FLOPs. Acceptance is based on end-to-end timing.

Production accepted: 8k median 37.54 ms versus r138 39.12 ms; 256k 39.03 ms / TTFT 73.691 s. Exact output/state/history checks, NLL 2.165267, 3/3 facts and cache checks passed. Service r139, grouped down disabled, manager active.
