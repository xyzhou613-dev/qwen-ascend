# r154 reproducible experimental evidence

See docs/r154-w8a8-gemv.md. These are online, bounded, non-exclusive diagnostic
measurements; none is a new production HTTP TPOT/TTFT result.

production-selected-config.json contains only whitelisted inference settings.
launch.py selects one visible device and a private candidate package. It was
created from unchanged r152 live source, adding engine.patch and w8a8.py. The
selected package under experiments uses selected-engine.patch and SmoothQuant.
launch_selected.py shows its calibration requirement and isolated module path.

All timing stages were sequential. Do not run multiple NPU probes concurrently.
Use container-internal timeout -k for bounded helpers; inspect current service,
memory headroom and ownership before reusing scripts. Never start a second Engine.

Frozen suite-inputs.json and calibration-inputs.json are canonical inputs.
prepare_suite.py/prepare_calibration.py document their construction; source files
were staged from the host into suite-sources, since /root/qwen-ascend is not
mounted inside the model container. Those source copies need not be regenerated
to rerun the saved token inputs. Calibration and development slices are separate.
calibrate.py streams one real layer at a time, storing only channel maxima.
sweep_suite.py and its variants stream all 48 layers, with their hidden sequences
on CPU. No Engine is constructed. Live-r152 parity was verified at 16/320 targets.

sweep-suite-full.json and sweep-suite-selective.json retain failed variants.
sweep-suite-smooth.json records the calibrated result. This is development
teacher-forced NLL, not independent task accuracy, code pass@1 or prefill quality.

gemv*_probe.py compare direct packed-W4 kernels, including BF16 output casting.
transpose_probe.py checks lossless full-expert layout changes and their cost.
loader_check.py checks selected constructor wiring by output/state hashes.
projection probes cover decode and prefill projection shapes, not a full prefill.

CPU reproduction: run analyze.py beside the JSON files; it writes summary.json
and report.md. AST checks, source/ELF hashes and real hardware results are retained.
SHA256SUMS is relative to this evidence directory.

No model weights, private-artifacts, per-token hidden activations, calibration
vectors, graph caches, full process environments or credentials are included.
