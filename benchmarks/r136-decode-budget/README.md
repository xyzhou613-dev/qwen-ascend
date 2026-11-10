# r136 decode-budget evidence (r135 runtime)

See ../../docs/r136-decode-budget.md. GB=10^9 bytes; main-memory PMU is not per-op HBM traffic.

Only summaries, active tensor inventory and analysis/calibration source are committed. Raw profiler CSVs, HBM sqlite and request events remain in /models/fnx-diagnostics/r136-decode-budget-20261110 and the delivered local evidence directory. For offline reproduction, copy the corresponding traces/, sys_hbm/, sys_request/ from the evidence into this directory, then run analyze.py, semantic_ops.py, resources.py and integrate_hbm.py with Python 3. No NPU is needed for aggregation.

calibrate_single.py is an optional NPU calibration, run separately per visible device after building libnop.so; it must not be confused with a kernel-independent additive overhead measurement. The original two-device calibration with context errors was excluded.

Production inference code is unchanged. Instrumented profiler entrypoints require the remote cloned fnx source; they are not standalone capture tools. No original process environment or credentials are included.
