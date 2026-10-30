"""★★ msProf / torch_npu.profiler 真实算子分布（第 34 轮）

目的：不再用消融推断，而是直接看设备侧时间到底花在哪些算子上。
做法：eager 与 captured 各采 3 步，导出 PROF 数据，按算子聚合 Task Duration。
"""
import csv
import glob
import os
import time

import torch
import torch_npu

from fnx import engine as E
from fnx.model import FnModel

OUT = "/tmp/prof_fnx"


def dump_summary(tag, outdir, n_top=16):
    files = glob.glob(os.path.join(outdir, "**", "*.csv"), recursive=True)
    kd = [f for f in files if "kernel_details" in os.path.basename(f)]
    op = [f for f in files if "op_summary" in os.path.basename(f)]
    use = (kd or op or files)
    if not use:
        print(f"  [{tag}] 没找到 CSV（{outdir}）", flush=True)
        return
    path = sorted(use)[-1]
    print(f"\n  [{tag}] {os.path.basename(path)}", flush=True)
    with open(path, encoding="utf-8", errors="ignore") as f:
        rd = csv.DictReader(f)
        cols = rd.fieldnames or []
        rows = list(rd)
    dur_col = next((c for c in cols if "Duration" in c or "duration" in c or "dur" in c.lower()), None)
    type_col = next((c for c in cols if c.strip() in ("OP Type", "Op Type", "OP_Type", "Type")), None)
    name_col = next((c for c in cols if c.strip() in ("Op Name", "OP Name", "Name", "Kernel Name")), None)
    if not dur_col:
        print(f"    ⚠️ 找不到时长列，列名: {cols[:10]}", flush=True)
        return
    tot = 0.0
    by_type, by_name = {}, {}
    for r in rows:
        try:
            d = float(r[dur_col])
        except (TypeError, ValueError):
            continue
        tot += d
        if type_col:
            by_type[r[type_col]] = by_type.get(r[type_col], 0.0) + d
        if name_col:
            nm = r[name_col][:58]
            by_name[nm] = by_name.get(nm, 0.0) + d
    print(f"    总设备时间 {tot/1e3:.2f} ms（{len(rows)} 条任务）", flush=True)
    if by_type:
        print(f"    ── 按 OP Type（Top {n_top}）──", flush=True)
        for k, v in sorted(by_type.items(), key=lambda x: -x[1])[:n_top]:
            print(f"      {v/1e3:8.2f} ms  {v/tot*100:5.1f}%   {k}", flush=True)
    if by_name:
        print(f"    ── 按 Op Name（Top {n_top}）──", flush=True)
        for k, v in sorted(by_name.items(), key=lambda x: -x[1])[:n_top]:
            print(f"      {v/1e3:8.2f} ms  {v/tot*100:5.1f}%   {k}", flush=True)


def profile(tag, m, capture_steps=3):
    exp = os.path.join(OUT, tag)
    os.makedirs(exp, exist_ok=True)
    acts = [torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU]
    torch.npu.synchronize()
    with torch_npu.profiler.profile(
            activities=acts,
            schedule=torch_npu.profiler.schedule(wait=0, warmup=1, active=capture_steps, repeat=1),
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(exp)) as prof:
        for _ in range(capture_steps + 1):
            m.step_pad(100, with_head=True)
            prof.step()
    torch.npu.synchronize()
    time.sleep(2)
    dump_summary(tag, exp)


def main():
    os.environ.setdefault("FNX_NO_CAPTURE", "0")
    t = time.time()
    m = FnModel(P=2048, lpc=45, dev0="npu:0", dev1="npu:1", nlayers=48,
                with_head=True, with_ngram=False, verbose=False)
    if os.environ.get("FNX_NO_CAPTURE") != "1":
        m.capture()
    print(f"  装配完成 {time.time()-t:.0f}s  capture={'有' if m.g0 else '无'}", flush=True)
    m.reset()
    for _ in range(2):
        m.step_pad(100, with_head=True)          # 暖机
    torch.npu.synchronize()
    tag = "captured" if m.g0 else "eager"
    profile(tag, m)
    print(f"\n  采集完成，原始数据在容器内 {OUT}/<tag>/", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
