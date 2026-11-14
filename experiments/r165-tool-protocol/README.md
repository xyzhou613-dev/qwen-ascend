# r165 tool protocol repair

Accepted on 192.168.18.206:9002. Model/engine r152, protocol r165.
No NPU/model math changes; only three HTTP protocol files changed/added.
39 pure tests + 28 real-tokenizer handler cases + 12 live interface cases pass,
plus typed tool-result roundtrip and enabled-thinking required-tool probes.

Run `python3 tests/t_tool_protocol.py` from the repository for the pure tests.
The actual-handler probe also needs the deployed Python dependencies and real
tokenizer at /models/Qwen3.8-Flash-Next. Use `FNX_TEST_RUNTIME=<repo>` and optionally
`FNX_TEST_OUTPUT=<output-dir>` with `python tests/t_server_tool_protocol.py`.
The model constructor is forbidden by that test; it does not create an Engine.

Controller paths here record this specific bounded rollout. Do not rerun the
old start script or reuse/reset its window. New maintenance requires a fresh
directory, current source ownership/hash checks and appropriate authorization.
Backup and independent rollback are preserved; accepted releases are protected
from accidental rollback. Raw logs/old source and hashes are in the archive;
repository log copies only trim trailing whitespace. Captured process env is
never published and was removed on acceptance.

See ../../docs/r165-tool-protocol.md and ../../benchmarks/r165-tool-protocol/.
