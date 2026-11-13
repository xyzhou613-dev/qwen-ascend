# r156 original-layout MoE down experiment

Research candidate only. Production remains r152; no whole-model TPOT or final quality acceptance for this candidate.

The selected implementation is the two-kernel path: direct K-major down + diagonal epilogue. `fused_down.cpp` is a rejected slower one-kernel variant. Its source and measured results are retained; `kmajor_down.merge` does not call it. `diag_epilogue.cpp` preserves the production epilogue rounding and sum order.

Build `bash build.sh` inside the existing Ascend CANN 9.1.0 environment. This uses a single compiler translation unit to avoid duplicate g_tilingKey symbols. The library ABI is for dav-c220-vec and the measured 910B4-1 environment.

The combined selected-engine.patch is against production r152. It includes the prior SmoothQuant constructor/dispatch changes plus the new decode-only MoE branch. Candidate packaging also needs `fnx/w8a8.py` and `fnx/smooth_w8a8.py` from experiments/r154-w8a8-gemv, and this module/library alongside them. Enable the prior SmoothQuant environment with its calibration directory, then explicitly set FNX_KMAJOR_DOWN=1. Defaults preserve the original path. This is not a production deployment instruction or quality approval.

Before the new MoE path, the activation and FP32 SwiGLU remain unchanged. The new path consumes cached BF16 scales and leaves all grouped prefill paths untouched. Complete prefill/decode and model quality remain to be validated. The abandoned install/apply_diag interception was bypassed by the production fused branch; the explicit selected-engine.patch now targets that actual branch.
