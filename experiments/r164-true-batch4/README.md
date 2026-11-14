# r164: experimental true B4 full-model decode

This is the recorded private fixed-batch measurement, not a deployed HTTP
continuous-batching scheduler. Production remains r152. Full quality testing
has not been completed for the changed numerical path.

The frozen initial harness and the supplemental ND writer are both retained.
`prepare_nd.py` generates readable `batch_ops_nd.py` and `bench_nd.py` from the
initial sources. The ND revision keeps three short cohorts and one long cohort,
respects both stop tokens, and uses chunk2048 for shared-prefix prefill.

The measured runtime was a private copy of the 45 canonical production files;
`prepare_runtime.py` enables the explicitly experimental B dimension guard.
The immutable source/flag checksums and complete result timelines are in
`../../benchmarks/r164-true-batch4/`. Recreate the private `runtime/fnx` and
`selected-config.json` in a NEW diagnostic directory, check the canonical hashes,
and obtain a NEW bounded exclusive window before any full model is loaded.
The paths in the recovery/controller scripts describe this particular run and
must be reviewed for a future machine/run. Never reuse the old authorization or
reset its deadline; the archived start marker deliberately prevents that.

The initial 256k scatter graph OOM and continuation-guard startup error are
preserved. Cross-device old-graph reset emitted EE1010 warnings; subsequent
fresh graph/eager checks passed, but the lifecycle issue remains a limitation.
No production engine, server scheduler, launcher, or model weights are changed.

See `../../docs/r164-true-batch4.md` for results, definitions and recovery proof.

Repository console extracts trim trailing whitespace; the evidence archive
retains the raw logs and their original hashes.
