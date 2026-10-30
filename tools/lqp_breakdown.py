"""★ LQP artifact 每层到底装了什么：expert 占多少、dense 占多少。
目的：判断"稠密权重 int8→int4"能不能腾出足够空间让 48 层全部进 dev0 图捕获
     （需要每层再省约 60 MB；48 层 × 1.31 GiB = 62.9 GiB > 60.96 上限）。

纯宿主脚本，只用 struct/json 读 safetensors 头，不加载权重。
"""
import json
import os
import struct

LQP = os.environ.get("FNX_LQP", "/models/Qwen3.8-Flash-Next-LQP")
QDIR = os.environ.get("FNX_QDIR", "/models/Qwen3.8-Flash-Next-Q")

DT = {"F32": 4, "F16": 2, "BF16": 2, "I8": 1, "U8": 1, "I16": 2, "I32": 4, "I64": 8,
      "F64": 8, "BOOL": 1, "U16": 2, "U32": 4}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode("utf-8")), 8 + n


def nbytes(meta):
    dt = DT.get(meta["dtype"], 0)
    cnt = 1
    for d in meta["shape"]:
        cnt *= d
    return cnt * dt, cnt


def categorize(name):
    n = name
    if "experts" in n:
        return "experts"
    if any(k in n for k in ("layernorm", "norm", "bias")):
        return "norms"
    if any(k in n for k in ("q_proj", "k_proj", "v_proj", "o_proj", "qkv", "attn",
                            "conv", "in_proj", "out_proj", "gate", "up_proj", "down_proj",
                            "mlp", "linear", "delta", "ssm", "a_proj", "b_proj")):
        return "dense"
    return "other"


def report(path, tag):
    hdr, base = header(path)
    hdr.pop("__metadata__", None)
    tot = 0
    by_cat = {}
    by_cat_cnt = {}
    per_suf = {}
    for name, meta in hdr.items():
        if name.endswith(".__meta"):
            continue
        b, cnt = nbytes(meta)
        cat = categorize(name)
        by_cat[cat] = by_cat.get(cat, 0) + b
        by_cat_cnt[cat] = by_cat_cnt.get(cat, 0) + cnt
        tot += b
        suf = name.rsplit(".", 1)[-1]
        per_suf[suf] = per_suf.get(suf, 0) + b
    print(f"\n=== {tag}: {os.path.basename(path)}  合计 {tot/2**30:.3f} GiB ===")
    for c in sorted(by_cat, key=lambda k: -by_cat[k]):
        print(f"  {c:8s} {by_cat[c]/2**20:9.1f} MiB  ({by_cat_cnt[c]/1e6:8.2f} M 参数)")
    for s in sorted(per_suf, key=lambda k: -per_suf[k]):
        print(f"    .{s:6s} {per_suf[s]/2**20:9.1f} MiB")
    return tot, by_cat


def main():
    files = sorted(f for f in os.listdir(LQP) if f.startswith("L") and f.endswith(".safetensors"))
    print(f"LQP 目录 {LQP}: {len(files)} 个层文件")
    if files:
        tot, by_cat = report(os.path.join(LQP, files[0]), "LQP 第 0 层")
        q = sorted(f for f in os.listdir(QDIR) if f.endswith(".safetensors"))
        print(f"\nQ 目录 {QDIR}: {len(q)} 个文件，示例：{q[:3]}")
        if q:
            report(os.path.join(QDIR, q[0]), "Q 目录（int8 容器）示例")
        print("\n=== 关键推算 ===")
        dense = by_cat.get("dense", 0)
        norms = by_cat.get("norms", 0)
        print(f"  每层 dense+norms = {(dense+norms)/2**20:.1f} MiB")
        print(f"  若 dense 从 int8 → int4，每层可省 ≈ {dense/2/2**20:.1f} MiB")
        print(f"  48 层总计可省 ≈ {dense/2*48/2**30:.2f} GiB")
        need = 48 * tot / 2**30 - 60.0
        print(f"  要让 48 层全进 dev0（上限 ~60.0 GiB）还差 {need:.2f} GiB"
              f"（每层需省 {need/48*2**20:.0f} MiB）")


if __name__ == "__main__":
    main()
