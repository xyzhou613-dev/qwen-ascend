# r155 SmoothQuant W8A8 whole-model TPOT

See [report](../../docs/r155-smooth-tpot-ab.md). Research evidence only; production remains r152.

The initial A completed. Candidate startup was interrupted by the overly strict 1e-5 numerical sanity check; the original attempt is preserved. After canonical restoration, the resumed controller reused A and completed fresh B/A2. The candidate smoke tolerance is 0.001 nat; this does not change the r154 quality limits or establish final quality acceptance.

Runtime scripts use the original absolute diagnostic directory. `controller.sh` is the resumed B/A2 controller; the original A/B/A version is in `initial-attempt/`. These scripts temporarily stop the service and must not be launched casually. They require a saved original service snapshot and the independent rollback mechanisms described in the report. `launch.py` refuses a second Engine.

Candidate code is the r154 selected SmoothQuant implementation at f9284fa (`experiments/r154-w8a8-gemv`) plus the two instrumentation patches here. Runtime Python/ELF hashes are recorded in runtime-code-manifest.json. Input payloads remain in the r152 evidence directory and are identified by input-manifest.json; copy the exact six files from benchmarks/r152-state-prefill/inputs when reproducing. Calibration vectors stay in the server private directory; calibration-reference.json records their hashes and r154 provides the frozen calibration inputs and collection code.

`http_bench.py` records fresh input and output counts, SSE DONE, server TPOT and HTTP TTFT. The fresh medians use three different fixed inputs with 128 generated tokens each. Cache and sustained generation are separate single runs; 1024 is the requested limit, not guaranteed actual output. `evaluate.py` regenerates summary.json; `report.py` renders the report from these summaries and quality-reference.json.

No model weights, calibration vectors, hidden states, complete process ENV, credentials or graph caches are included.
