# r151 DeltaNet prefill output assembly

Accepted only after exactness and end-to-end TTFT gates. Read report.md, ab-summary.json and production-verification.json. Three fresh 8k requests, two distinct fresh 262016+128 long requests and a nonstream cached repeat per version. Five frozen documents/995 targets use the real prefill path. All 36 real-weight DeltaNet layer subpaths and four complete layers were checked, including FP32 state bits.

Implementation uses FP32 addition followed by BF16 RINT conversion and writes token-major output directly. BMM and FP32 recurrence stay unchanged. The default kernel gathers eight heads per task, fixed [48,64,128] inputs; unsupported shapes fall back. Five benchmark layouts are included in the build, while production selects mode4. Linux aarch64 ELF must be paired with this source/CANN build. See implementation/ and kernels/prefill_output/.

Initial state Mul/Add fusion and output Add/Cast with a final cat were not selected. Their standalone prototype source, wrapper and library are kept in rejected-state-and-cat/. This does not enable those paths in production.

Operational scripts are historical and require the fixed server paths and a maintenance window. Full process ENV, credentials, model weights and graph caches are excluded. SHA256SUMS covers distributed files.
