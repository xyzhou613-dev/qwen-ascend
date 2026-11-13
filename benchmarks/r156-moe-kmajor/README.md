# r156 evidence

See [report](../../docs/r156-moe-kmajor.md). Real component/layer timings were collected before the 17:02 host reboot. They are not new HTTP TPOT values. After reboot, physical node indices changed and the original container could not start; recovery-device-change.json and final-operations.json record the authorized restoration. Production source and p275 hashes were unchanged. Rebuild performance baselines before further full-model comparisons.

`probe.py` tests the down projection including the legacy standalone diagonal extraction baseline. `merge_probe.py` measures the correct production fused baseline versus selected split and rejected fused candidates. `layer_probe.py` compares SmoothQuant alone versus the combination on four real layers. Every tested layer's dispatch counter increased by 33; see dispatch-audit.json. The wrapper now additionally asserts each layer's 16 pre-timing calls individually. The candidate library/module is in experiments/r156-moe-kmajor.

`fetch_quality.ps1`, freeze_quality.py and freeze_nll.py prepare immutable held-out inputs. Inputs contain 1100 tasks and 100352 scored NLL targets; no candidate quality run has completed. Official source revisions and SHA hashes are recorded. Private dataset snapshots, question/answer text, canonical code/tests and token IDs are retained on the server, not redistributed here. Re-fetch the pinned sources to reproduce. The source pool for MMLU-Pro covers 12 categories, not its entire official test set.

`quality_eval.py` resumes frozen input hashes and grades code only in a network-disabled/read-only/nonroot resource-limited temporary CPU container. Framework selftests passed. quality-r152screen contains only 32 MCQ/JSON baseline questions, not the full suite. The 100k NLL input is benchmark knowledge/math/code text; natural-text and true long-prefill acceptance remain outstanding.

`recover.sh` is a site-specific historical recovery script. It stops the Docker daemon only after confirming that every container is stopped, backs up the existing container configuration privately, and changes only the two missing device mappings. It does not recreate the container. Do not treat it as an ordinary experiment launcher. Docker raw configs and credential-bearing helpers are excluded.

No model weights, calibration vectors, hidden states, complete process ENV, graph caches, dataset copies or Docker raw configuration backups are included.
