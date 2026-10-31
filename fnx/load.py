"""权重装配：从 Q 目录流式加载单层；稠密一次性反量化，专家保留 int8。
另含 n-gram 表（≈51 GB，驻留磁盘/page cache）的 mmap 随机行读取。
"""
import json
import os
import struct

import numpy as np
import torch

from . import engine as E

# 路径支持环境变量覆盖：移植到别的机器时权重往往不在 /models（例如非 root 用户建不了 /models）。
QDIR = os.environ.get("FNX_QDIR", "/models/Qwen3.8-Flash-Next-Q")
QPACKED = os.environ.get("FNX_QPACKED", "/models/Qwen3.8-Flash-Next-QP")   # tools/repack_q4.py 产出的真 4-bit 打包专家权重
SRCDIR = os.environ.get("FNX_SRCDIR", "/models/Qwen3.8-Flash-Next")
PREFIX = "model.language_model.layers."


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n).decode("utf-8"))
    return hdr, 8 + n


def _group(sd):
    out = {}
    for k, v in sd.items():
        if k.endswith((".__pad", ".__shape")):
            out.setdefault(k.rsplit(".", 1)[0], {})[k.rsplit(".", 1)[1]] = v
        elif k.endswith((".q4", ".s4", ".q8", ".s8")):
            base, suf = k.rsplit(".", 1)
            out.setdefault(base, {})[suf] = v
        else:
            out[k] = {"raw": v}
    return out


def load_layer(path, layer_idx, device):
    """返回 {短名: Raw | ExpQ | Tensor}，全部已迁到 device。"""
    from safetensors.torch import load_file
    sd = load_file(path)
    g = _group(sd)
    pre = f"{PREFIX}{layer_idx}."
    res = {}
    for base, parts in g.items():
        name = base[len(pre):] if base.startswith(pre) else base
        if "raw" in parts:
            res[name] = parts["raw"].to(device)
            continue
        if "q4" in parts:
            # ★ 打包 torch 路径：q **不进显存**，直接在宿主打好包再传 qv2（1.17 GiB）。
            # 否则设备上要同时放 int8 容器(2.34) + qv2(1.17)，峰值多 1.56 GiB/层，
            # 单卡只能装 37 层（实测 OOM）；宿主打包后可装 44 层。
            _packed = os.environ.get("FNX_PACKED_TORCH") == "1"
            # ★★ 第 37 轮：aclnn per-group W4A16 路径（默认关）。
            #   离线 repack 出来的是 [K,E,N/8] int32（int4pack）+ [G,E,N] int8 scale
            #   + [E,N] bf16 base，见 tools/repack_w4a16.py。
            _w4dir = os.environ.get("FNX_W4A16_DIR",
                                    "/models/Qwen3.8-Flash-Next-LQP-w4a16")
            _w4f = os.path.join(_w4dir, f"L{layer_idx:02d}.w4.safetensors")
            if os.environ.get("FNX_MOE_W4A16") == "1" and os.path.exists(_w4f):
                from safetensors.torch import load_file as _lf
                _w = _lf(_w4f)
                try:
                    res[name] = E.ExpW4(_w[name + ".w4"].to(device),
                                        _w[name + ".s8t"].to(device),
                                        _w[name + ".base"].to(device), 64)
                finally:
                    del _w
                continue
            # ★ 优先读离线 artifact（LQP）：v2 + offset-binary 打包 + 尺度都在里面，
            # 跳过 merge_scales_g64 与运行时打包 → 冷启动从 13.6 分钟回到 ~7.6 分钟。
            _lqp = os.path.join(os.environ.get("FNX_LQP",
                                               "/models/Qwen3.8-Flash-Next-LQP"),
                                f"L{layer_idx:02d}.safetensors")
            if _packed and os.path.exists(_lqp):
                from safetensors.torch import load_file
                _a = load_file(_lqp)
                _k = name          # 已经是 "mlp.experts.<proj>"，不要再加前缀
                try:
                    res[name] = E.ExpQ.from_packed(_a[_k + ".v2o"],
                                                   _a[_k + ".s8"],
                                                   _a[_k + ".base"], device)
                finally:
                    del _a
                continue
            qh = parts["q4"]                       # 宿主容器
            q = qh if _packed else qh.to(device)
            s = parts["s4"].to(device)
            pk = os.environ.get("FNX_PACK", "0") == "1"
            if os.environ.get("FNX_EXP_G", "32") == "64":   # 在宿主内存里合并，设备峰值只留结果
                gs_in = q.shape[-1] // s.shape[-1]
                qc_, sc_ = parts["q4"], parts["s4"]
                gs_in = qc_.shape[-1] // sc_.shape[-1]
                q2, s8, base = E.merge_scales_g64(qc_, sc_, gs_in=gs_in, gs_out=64)
                s8 = s8.to(device)
                base = base.to(device)
                if _packed:
                    eq = E.ExpQ(q2, s8, 64, packed=False, base=base)   # q2 在宿主
                    del q2
                else:
                    # 分块上卡：瞬时只用 ~0.2 GiB，而不是一次性 1.56 GiB（24 层满载时后者会把卡打爆）
                    q = torch.empty(q2.shape, dtype=torch.int8, device=device)
                    for e0 in range(0, q2.shape[0], 64):
                        q[e0:e0 + 64] = q2[e0:e0 + 64].to(device)
                    del q2
                    eq = E.ExpQ(q, s8, 64, packed=False, base=base)
                eq.base = base
                res[name] = eq
            else:
                res[name] = E.ExpQ(q, s, q.shape[-1] // s.shape[-1], packed=pk)
            continue
        q = parts["q8"].to(device)
        s = parts["s8"].to(device)
        pad = int(parts["__pad"].item()) if "__pad" in parts else 0
        if q.dim() != 2:
            # conv1d 这类小张量直接反量化成 bf16
            w = E.deq(q, s)
            del q, s
            if pad:
                w = w[..., : w.shape[-1] - pad].contiguous()
            res[name] = w
        else:
            # 稠密线性层：int8 常驻。
            # ★ 第 38 轮：`FNX_DENSE_W8=1` 时走 aclnn W8 per-group（Cube 内反量化），
            #   实测 0.953 → 0.263 ms/层，数值与旧路径**逐位一致**，且不占 bf16 副本。
            kk = q.shape[-1] - pad
            if os.environ.get("FNX_DENSE_W8") == "1":
                gs_ = q.shape[-1] // s.shape[-1]
                if kk % gs_ == 0:
                    try:
                        res[name] = E.DenseW8(q, s, K=kk)
                        continue
                    except Exception as _e:  # noqa: BLE001
                        print(f"[dense_w8] {name} 退回 QW: {_e}", flush=True)
            res[name] = E.QW(q, s, K=kk)
    del sd, g
    return res


def load_nonlayer(embed_dev, head_dev, with_lmhead=True):
    """只取 embed_tokens / lm_head / hyper_connection_mixer（视觉塔不加载）。

    显存账（48 层要 128.2 GiB，只有 121.2 可用，必须省）：
      * embed_tokens 只做 1 行 gather → 留在**宿主 DDR**，每 token 送 5 KB，省 0.59 GiB
      * lm_head / 层内稠密 → int8 常驻 + 调用时反量化（QW），省 4.04 GiB
    """
    from safetensors import safe_open
    p = os.path.join(QDIR, "nonlayer.safetensors")
    mx = {}
    with safe_open(p, framework="pt", device="cpu") as f:
        eq = f.get_tensor("model.language_model.embed_tokens.weight.q8")
        es = f.get_tensor("model.language_model.embed_tokens.weight.s8")
        if with_lmhead:
            lq = f.get_tensor("lm_head.weight.q8")            # 常驻宿主 DDR
            ls = f.get_tensor("lm_head.weight.s8")
        else:
            lq = ls = None
        try:
            lpad = int(f.get_tensor("lm_head.weight.__pad").item())
        except Exception:
            lpad = 0
        prefix = "model.language_model.hyper_connection_mixer."
        for n in ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight"):
            k = prefix + n
            base = k[:-len(".weight")] if n.endswith(".weight") else k
            if k + ".q8" in f.keys():
                q = f.get_tensor(k + ".q8").to(head_dev)
                s = f.get_tensor(k + ".s8").to(head_dev)
                try:
                    pad = int(f.get_tensor(k + ".__pad").item())
                except Exception:
                    pad = 0
                mx[n] = E.QW(q, s, K=q.shape[-1] - pad)
            else:
                mx[n] = f.get_tensor(k).to(head_dev)
    if with_lmhead:
        # ★ 第 39 轮：`FNX_DENSE_W8=1` 时 lm_head 也走 aclnn W8（int8 常驻，读的字节腰斩）。
        #   算子要求 N ≤ 65535，而词表 248320 → 必须按 N 预切 4 段（62080 each）。
        #   前提是 `FNX_DEQ_HEAD=0`，否则会被装载期预反量化成 1.27 GB 的 bf16。
        lw = None
        if os.environ.get("FNX_DENSE_W8") == "1":
            try:
                lw = E.DenseW8(lq.to(head_dev), ls.to(head_dev), K=lq.shape[-1] - lpad)
                lw.make_chunks(62080)
                print(f"[dense_w8] lm_head → int8 分 {len(lw.chunks)} 段，"
                      f"每段 N≤62080（算子上限 65535）", flush=True)
            except Exception as _e:  # noqa: BLE001
                print(f"[dense_w8] lm_head 退回 QW: {_e}", flush=True)
                lw = None
        if lw is None:
            lw = E.QW(lq.to(head_dev), ls.to(head_dev), K=lq.shape[-1] - lpad)
            lw.chunk_n = 16384                      # 必须分块：整块反量化要 1.19 GiB
        #   宿主算 lm_head 实测 1600 ms/token，否决
    else:
        lw = torch.zeros(1, 1, dtype=torch.bfloat16, device=head_dev)
    return (eq, es), lw, mx


class NgramTable:
    """51 GB n-gram 表（INT8，128 shard）驻留磁盘；每 token 只读 16 行（≈2.5 KB）。"""

    def __init__(self, qdir=QDIR, nshard=128):
        shards = []
        for i in range(nshard):
            p = os.path.join(qdir, f"NG{i:03d}.safetensors")
            hdr, base = read_header(p)
            kq = [k for k in hdr if k.endswith(".q8") and k != "__metadata__"][0]
            ks = kq[:-3] + ".s8"
            rows, cols = (int(x) for x in hdr[kq]["shape"][:2])
            assert cols == E.NG_DIM, (p, cols)
            mm = np.memmap(p, dtype=np.uint8, mode="r")
            shards.append(dict(
                q_off=base + hdr[kq]["data_offsets"][0],
                s_off=base + hdr[ks]["data_offsets"][0],
                rows=rows, mm=mm))
        self.shards = shards
        self.cum = np.cumsum([s["rows"] for s in shards])
        self.total = int(self.cum[-1])
        self.gs = E.NG_DIM // 32
        self.s_bytes = self.gs * 2
        self.per = E.NG_DIM // self.gs

    def row(self, gid):
        i = int(np.searchsorted(self.cum, gid, side="right"))
        prev = int(self.cum[i - 1]) if i else 0
        sh = self.shards[i]
        local = int(gid) - prev
        q = np.frombuffer(sh["mm"], dtype=np.int8, count=E.NG_DIM,
                          offset=sh["q_off"] + local * E.NG_DIM)
        sc = np.frombuffer(sh["mm"], dtype=np.float16, count=self.gs,
                           offset=sh["s_off"] + local * self.s_bytes)
        return q.astype(np.float32) * np.repeat(sc.astype(np.float32), self.per)

    def gather(self, ids):
        out = np.empty((len(ids), E.NG_DIM), dtype=np.float32)
        for i, gid in enumerate(ids):
            out[i] = self.row(int(gid))
        return torch.from_numpy(out.reshape(1, 1, E.NG_HEADS * E.NG_DIM)).to(torch.bfloat16)


def load_packed_experts(layer_idx, device):
    """读真 4-bit 打包专家权重（供 W4A16 kernel 用）。

    返回 {which: (p4:int8 [E,N,K/2], s4:fp16 [E,N,K/gs])}，位语义见 QPACKED/manifest.json：
      byte j 低 4 bit = W[k=2j]，高 4 bit = W[k=2j+1]；value = ((nibble ^ 8) - 8)，范围 -8..7
    以 int8 承载保证 torch_npu 位运算可用性（数值解释权在 kernel）。
    """
    from safetensors.torch import load_file
    fp = os.path.join(QPACKED, f"L{layer_idx:02d}.packed.safetensors")
    sd = load_file(fp)
    out = {}
    for which in ("gate_up_proj", "down_proj"):
        k = f"model.language_model.layers.{layer_idx}.mlp.experts.{which}."
        out[which] = (sd[k + "p4"].view(torch.int8).to(device), sd[k + "s4"].to(device))
    return out


def packed_available(layer_idx=0):
    return os.path.exists(os.path.join(QPACKED, f"L{layer_idx:02d}.packed.safetensors"))