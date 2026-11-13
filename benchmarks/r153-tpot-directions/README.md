# r153 TPOT direction probes (not a deployed runtime)

See docs/r153-tpot-directions.md. Production stayed r152 throughout.

Scripts were executed inside bole-model-Flash with /root/ascend-dev/venv/bin/python.
Only launch.py selects production's whitelisted non-secret settings and one visible device;
hccl_verified_probe.py controls its own two ranks and unique TCP rendezvous port 29853.
All NPU timing runs were sequential, online and non-exclusive. Do not overlap them.
The scripts deliberately guard r152 and free HBM; inspect current ownership, versions,
memory headroom and process state before any future reuse. Never load another Engine.

CPU analysis: python analyze.py (expects the adjacent JSON files).
To reproduce NZ format creation, use an isolated private dependency directory; successful
probes used decorator, scipy, attrs and psutil plus ge_compat/sitecustomize.py for compiler
subprocess NumPy aliases. A SciPy/NumPy version warning remained. This is diagnostic
compatibility, not an approved production dependency change. Dependencies and pip logs
are deliberately excluded from the evidence package. No global pip operation was run.

dense_probe.py: hot single-weight ND layout comparison.
dense_stream_probe.py: >=384 MiB rotating-weight streaming proxy (ND).
dense_stream_nz_probe.py: same streaming proxy with verified format-29 replicas.
hccl_probe.py: initial timing; hccl_verified_probe.py adds fresh nonzero graph-sum verification.
verify_layer_probe.py: initial generic-512-group allocator-cap failure, retained as negative evidence.
verify_layer_small_probe.py: active/padded expert groups, graph-safe and bounded.
verify_layer_loop_probe.py: tokenwise MoE with batched attention/HC.
verify_layer_native_probe.py + small_delta.py: T<=4 native FP32 recurrence with prefix-state journal.

Synthetic inputs, four sampled real layers and component timing do not constitute model
quality or end-to-end TPOT acceptance. No model weights, user activations, full process
environment snapshots, graph caches, private dependency binaries or credentials included.
