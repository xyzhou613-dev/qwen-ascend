# Ordered DeltaNet tiling: screened, not enabled

128 dynamic steps passed bitwise FP32 state/output comparison for row tiles 4/8/16/32, each with value-column widths 128 and 64. Elementwise work is blocked while both reductions retain their original 128-row accumulation order.

The initial screening ran with r140 resident and idle: reference median about31.11us, best 32x128 about30.39us; 64-column variants were slower. This is only a component screening, not a full-model speedup or controlled production acceptance. The gain was too small to justify deployment; production delta_native.py and libfnx_delta.so remain unchanged.

A first invalid harness cached the raw stream handle before asynchronous input preparation; a synchronized diagnostic and a per-call current-stream harness both passed. That invalid attempt is excluded from performance evidence. The public probe uses current-stream acquisition for each call, matching the production wrapper.

Build this directory with build.sh (output libtiled.so) and supply FNX_DN_REFERENCE_SO pointing to the installed reference library if not built under fnx/. No model weights, raw process environment or credentials are included.
