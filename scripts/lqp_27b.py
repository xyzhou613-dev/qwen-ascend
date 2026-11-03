"""Build and validate Qwen3.8-27B's BF16 vLLM TP=2 checkpoint."""

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import tempfile


TP = 2
FORMAT = "qwen38-27b-bf16-tp2-v1"


def file_stamps(files):
    return {
        file.name: [file.stat().st_size, file.stat().st_mtime_ns]
        for file in sorted(files)
    }


def source_stamp(source):
    index_path = source / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    shards = {source / name for name in index["weight_map"].values()}
    return file_stamps(shards | {index_path, source / "config.json"})


def metadata_files(source):
    return [
        file for file in source.iterdir()
        if file.is_file()
        and file.suffix not in {".safetensors", ".bin", ".pt"}
        and file.name != "model.safetensors.index.json"
    ]


def versions():
    return {
        name: importlib.metadata.version(name)
        for name in ("vllm", "vllm-ascend", "torch", "torch-npu")
    }


def shard_stamp(path):
    files = []
    for rank in range(TP):
        rank_files = list(path.glob(f"model-rank-{rank}-part-*.safetensors"))
        if not rank_files or any(f.stat().st_size < 16 for f in rank_files):
            raise ValueError(f"missing or empty rank {rank} checkpoint")
        files.extend(rank_files)
    return file_stamps(files)


def check(source, target, tp):
    manifest = json.loads((target / "lqp-manifest.json").read_text())
    if tp != TP or manifest != {
        "format": FORMAT,
        "source": str(source),
        "source_files": source_stamp(source),
        "software": versions(),
        "shards": shard_stamp(target),
    }:
        raise ValueError("LQP manifest does not match the source, shards or runtime")
    metadata = metadata_files(source)
    if file_stamps(metadata) != file_stamps(target / file.name for file in metadata):
        raise ValueError("LQP model metadata differs from the original directory")


def build(source, target):
    from vllm import LLM

    if target.exists():
        raise FileExistsError(target)
    source_files = source_stamp(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as tmp:
        stage = Path(tmp) / "checkpoint"
        stage.mkdir()
        os.environ["QWEN27_LQP_EXPORT_DIR"] = str(stage)
        llm = LLM(
            model=str(source),
            tensor_parallel_size=TP,
            dtype="bfloat16",
            enforce_eager=True,
            trust_remote_code=True,
            limit_mm_per_prompt={"image": 0, "video": 0},
            max_model_len=1024,
            gpu_memory_utilization=0.65,
            worker_cls="lqp_27b_worker.ExportWorker",
        )
        for file in metadata_files(source):
            shutil.copy2(file, stage / file.name)
        if source_files != source_stamp(source):
            raise ValueError("source checkpoint changed during export")
        manifest = {
            "format": FORMAT,
            "source": str(source),
            "source_files": source_files,
            "software": versions(),
            "shards": shard_stamp(stage),
        }
        (stage / "lqp-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        check(source, stage, TP)
        stage.rename(target)
        print(f"LQP checkpoint published at {target}", flush=True)
        llm.llm_engine.engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check"))
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--tp", type=int, default=TP)
    args = parser.parse_args()
    source, target = args.source.resolve(), args.target.resolve()
    if args.action == "build":
        if args.tp != TP:
            parser.error(f"export requires TP={TP}")
        build(source, target)
    else:
        try:
            check(source, target, args.tp)
        except (OSError, ValueError, KeyError) as exc:
            parser.exit(1, f"LQP check failed: {exc}\n")


if __name__ == "__main__":
    main()
