# Reboot A/B of r146 normalization candidate (r148 experiment)

The candidate is faster but rejected: 68 target NLL differences in the 12k fixture and one changed 8k answer. Production was restored to r144. Read report.md, ab-summary.json and verification.json. This commit does not activate or modify production runtime code.

Baseline and candidate each rebooted with the same production configuration, on loopback port 19002 without user traffic. Three frozen fresh 8k inputs, one near-capacity 256k input and a nonstream cached repeat per version. Quality uses three frozen documents and 597 targets through the true prefill policy. Sample counts and scope limits are explicit.

rejected-candidate-implementation.tgz includes the unaccepted C++ kernel, Python wrapper, Linux aarch64 library and engine diff. Do not install it as an exact optimization. Build settings are the same CANN9.1 Bisheng dav-c220-vec -O2 pattern as kernels/l2norm_post/build.sh, with norm_fused.cpp and an explicit output library path. Numeric implementation was unchanged from the r146/r147 component checks.

rounding-sparse-case-*.pt are small synthetic tensors rechecked at original [1,16,128] shape, query scaling disabled. The raw inherited silu field is inapplicable for this sparse generator; use rounding-reproducer-summary.json and the source generator. No user request activations were captured. No weights, graph caches, credentials or full process ENV are included.

Controller and restore scripts are historical operational evidence and depend on fixed server paths plus a fresh private snapshot. The private snapshot was removed after recovery. Do not run the controller on an online service without a maintenance window and renewed site checks. SHA256SUMS covers all distributed files.
