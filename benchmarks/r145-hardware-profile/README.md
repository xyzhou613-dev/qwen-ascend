# r144 dual-process timeline and hardware counters (r145 capture)

Service implementation: 794a22e (r144). Sample date: 2026-11-12. Two Ascend 910B4-1 cards, physical 3/5, logical 0/1. This dataset does not change the production implementation or numerical path.

Read report.md first. Timing, PipeUtilization, Memory, MemoryAccess, L2Cache and system hardware were collected in separate passes. Do not use PMU-pass latency as normal TPOT or add overlapping kernel-duration categories. Timing baseline is direct Engine.generate, not HTTP/SSE. CPU wait spans include device execution.

## Files and units

- summary.json: closed interval unions, weighted counters, logical traffic, physical HBM integration, units and limitations.
- runs.json / clock-alignment.json / *_host[01].json: token positions, host epoch/monotonic timestamps and shared sequence numbers.
- aligned/*_two_tokens.json: ready-to-open Chrome Trace with two consecutive tokens and process labels.
- aligned/*_timing.json.gz: gzip-compressed complete Chrome Trace; decompress before opening in Perfetto/Chrome tracing. No artificial timestamp shift or scale was applied.
- aligned/*_system_counters_full.csv: every exported HBM/PCIe counter point, including generation-window flanks. timestamp_ns is epoch ns. Throughput is MB/s; PCIe nonpost_latency is us. HBM totals sum four channels/card. The original hbm.csv Average is a channel average.
- raw-kernel-task-csv.tgz: all 20 stage×metric kernel_details.csv and task_time.csv pairs; original relative paths retained.
- raw-system-json.tgz: both original msprof system JSON exports (includes startup/idle, excluded from decode integration).
- system-summaries/: original HBM/PCIe summary CSVs. These cover the tool window, not just the decode window.
- coverage-audit.json: constant per-period kernel count/signature checks for every metric pass. The profiler emitted RECORD-state shutdown warnings; coverage checks do not prove every unsupported field is valid.
- public-config.json / final-validation.json / *health.json / post-restore-http.json: selected nonsecret configuration and canonical service restoration evidence.
- unchanged-production-manifest.json: production Python/ELF digests compared to the initial snapshot; diagnostic pipeline files excluded and checked against repository separately.

All direct Memory-group L2 read/write GB/s fields returned zero; physical L2 bandwidth is unavailable. L2Cache request hit/miss counts and MemoryAccess logical GM data are valid separate observations. Logical KB uses 1024 bytes, converted to GiB; physical HBM MB/GB uses decimal units. PCIe throughput zero samples cannot establish zero traffic or a utilization percentage. MAC/Vector cycle ratios are not peak FLOPS attainment.

## Reproduce the analysis without an NPU

Extract raw-kernel-task-csv.tgz and raw-system-json.tgz into this directory, preserving paths. Run python analyze.py, python audit.py, python finalize.py. plot.py also needs matplotlib; install it into an isolated plotting venv. These analysis commands never load an Engine. The full system JSON and traces can be reprocessed from the exported counter/task timestamps.

The capture/controller/recovery scripts are historical operational evidence. Do not execute controller.sh as an unattended rerun recipe: it is the 256k resume controller, depends on a live private snapshot and fixed server paths, and deliberately stops/restores services. The private full ENV snapshot was removed after recovery and is not distributed. Recollection requires rechecking the current service, exclusive Engine ownership and recovery plan. Original prompts, all raw profiler databases and unchanged diagnostic model files remain only in /models/fnx-diagnostics/r145-hardware-profile-20261112; prompts are identified by token counts and input ID SHA256 in the output records.

The first capture finished 8k and failed at the long-prompt filename; production was automatically recovered. The second capture resumed only 256k. Both output prefixes match the accepted r144 reference. Canonical p275.sh, manager and one normal watchdog have been restored. No runtime optimization is claimed by this profiling-only commit.
