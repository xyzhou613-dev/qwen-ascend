"""fnx —— Qwen3.8-Flash-Next 昇腾专用后端（graph-safe、单 token 步进）。

设计要点（均由前 24 轮实测决定）：
  * 不做 TP/EP：单卡放 24 层（PP=2），卡间每 token 仅传 20 KB。
  * forward 天生 graph-safe：无 .item()/.all()、无依赖张量值的分支、形状全静态。
  * prefill = 逐 token 复用同一个 step（与 decode 同一条代码路径，天然一致）。
  * 稠密 W8 在装配时一次性反量化为 bf16 常驻；专家 W4 常驻 int8，每 token 只 gather top-10。
  * QSA 稀疏选块（indexer）暂未实现 —— 上下文 ≤ indexer_budget(2048) 时稠密 causal 与参考实现**逐位一致**。
"""
import math
import os
import numpy as np
import torch
import numpy as np
import torch.nn.functional as F
try:                       # 融合 RMSNorm 需要；缺了自动退回纯 torch 实现
    import torch_npu  # noqa: F401
except Exception:          # noqa: BLE001
    torch_npu = None

# ---------------- 结构常量（config.json / text_config） ----------------
H = 2560
HC = 4
HCH = H * HC                      # 10240
LOWRANK = 320

NKH = 16
NVH = 48
HKD = 128
HVD = 128
KEY_DIM = NKH * HKD               # 2048
VAL_DIM = NVH * HVD               # 6144
CONV_DIM = KEY_DIM * 2 + VAL_DIM  # 10240

NAH = 24
KVH = 2
HD = 256

NEXP = 512
TOPK = 10
MI = 640

VOCAB = 248320
EPS = 1e-6

ROPE_DIM = 64
ROPE_HALF = ROPE_DIM // 2
ROPE_THETA = 10_000_000.0

NG_HEADS = 16
NG_DIM = 160
NG_SIZE = 3
NG_EOS = 248044
NG_CTX = NG_SIZE - 1

PLE_LAYER = 1
PLE_CONV_LEN = (4 - 1) * NG_SIZE  # 9

NEG = -1.0e30
FULL_TYPES = ("full_attention", "indexed_attention")

# ★★ 采样批大小（第31轮）：一次前向处理 B 条**独立**序列，用来摊薄
#   "每 token 都要把整层 int8 权重反量化一遍"的固定成本。
#   微基准（tests/t_batch_micro.py）：int8 权重单次调用耗时 **与 B 无关**
#   （B=1 0.20 ms / B=8 0.19 ms / B=32 0.19 ms），因为成本全在反量化那一次，
#   而 B=1 时后续 GEMM 几乎是免费的（纯 bf16 matmul 只 0.04 ms）。
#   ⇒ B=8 时稠密权重成本可摊到 1/8。
#   注意：**图捕获按固定 B**，所以 B 在进程生命周期内不可变；B=1 时行为与改造前完全一致。
B = int(os.environ.get("FNX_BATCH", "1"))

# ★★ 装载期预反量化（第31轮提速主线）：
#   实测（第 30 轮 §16 反量化地板 + tests/t_batch_micro.py）：int8 权重一次调用 0.20 ms，
#   其中纯 bf16 matmul 只占 0.04 ms ⇒ **0.16 ms 是"把这块 int8 反量化成 bf16"的固定开销，
#   与权重多大无关**。所以每层 11 个小权重（mixer / shared-expert / gate，只占稠密字节 26%）
#   吃掉了 69% 的稠密开销 —— 它们最亏。
#   做法：装载期一次性 `deq_w()` 成 bf16 常驻（`Raw`），每 token 直接 GEMM。
#   数值与原路径**逐位相同**（反量化本来就只是 q×scale），只是不再每 token 重算。
#   显存代价：小权重 +10 MiB/层；lm_head +0.6 GiB（放在 dev1，那里空得很）。
_DEQ_SMALL = os.environ.get("FNX_DEQ_SMALL", "0") == "1"
_DEQ_HEAD = os.environ.get("FNX_DEQ_HEAD", "0") == "1"
#   ★ 注意力投影（w_qkv / w_z / w_out，int8 共 55 MiB/层）是剩下最大的一块反量化开销，
#   但反量化成 bf16 要 +55 MiB/层 —— dev0 没这么多余量。所以按"层数预算"开放：
#   `FNX_DEQ_ATTN=<n>` = dev0 上前 n 层做预反量化（dev1 的层永远做，那里显存富余）。
_DEQ_ATTN = int(os.environ.get("FNX_DEQ_ATTN", "0"))
#   scale 组从 g64 翻到 g128 可省 18.8 MiB/层（45 层约 0.85 GiB），
#   够再预反量化 ~15 层注意力投影（~11 ms）。先量质量代价：`FNX_S8_SIM128=1`
#   在解包时把相邻两个 g64 scale 合并（等效 g128），只影响数值、不改磁盘。
_S8_SIM128 = os.environ.get("FNX_S8_SIM128", "0") == "1"
#   ★ 专家 3-bit 可行性实验：把 4-bit 的 q 量化到偶数（= 同动态范围下 8 个电平），
#   即精确模拟"专家权重降到 3-bit"的误差。能省 300 MiB/层（45 层 ≈ 13 GiB），
#   够把 48 层全部图捕获 + 全部注意力预反量化（潜在 ~70-80 ms）。
_SIM3 = os.environ.get("FNX_SIM_EXP3", "0") == "1"
#   ★ 第 34 轮按 msProf 的优化：把 offset-binary 的 `-8` 折进 scale 那一次乘
#   （`(n'-8)*s = addcmul(-8s, n', s)`）。
#   ⚠️ **默认关闭**：实测这条路会让单次 /fnx/loss 从 ~30 s 拖到 >105 s 仍未完成，
#   怀疑 `addcmul` 在 [k,N,S,1]×[k,N,S,64] 这种广播形态上走了慢路径（待查）。
#   保留代码与开关，方便以后用 msProf 单独看这个 op；生产用 0。
_FOLD8 = os.environ.get("FNX_DEQ_FOLD8", "0") == "1"
#   第 36 轮：4-bit 重新量化的裁剪系数（仅 repack 时生效）。1.0 = 不裁剪（基线）。
_Q4_CLIP = float(os.environ.get("FNX_Q4_CLIP", "1.0"))
#   ★ 第 38 轮：把"共用同一个输入"的稠密投影沿 N 拼成一个权重（qkv+z、b+a），
#   一次算子调用全出，层内再切片。动机是 profile 里 WeightQuantBatchmatmul 变成
#   最大单项（16 ms/步 / 315 次调用，平均 0.05 ms —— 成本在发射不在算）。
_DENSE_CAT = os.environ.get("FNX_DENSE_CAT", "1") == "1"
#   ★ 第 39 轮：bf16 小权重里"共用输入"的也拼成一次 GEMM
#   （mixer 的 down+inj、shared-expert 的 gate+up）。数值逐位不变。
#   `FNX_SMALL_MERGE`: 0=不拼, 1(或 all)=都拼, mix=只拼 mixer, sh=只拼 shared-expert
# ★ 第 46 轮：分块 prefill 的块大小 T（1 = 关闭，走原来的逐 token 路径）
_CHUNK = max(1, int(os.environ.get("FNX_CHUNK", "1")))
_CAT4 = os.environ.get("FNX_DENSE_CAT4", "0") == "1"       # 第 42 轮实测：4 合 1 不赚，默认关
_DN_RMS = os.environ.get("FNX_DN_RMS", "1") == "1"       # DeltaNet 输出归一化用融合 rms_norm
_INJECT_CM = os.environ.get("FNX_INJECT_CM", "0") == "1"   # 第 41 轮实测：addcmul 反而慢 ~0.5 ms，默认关
# ★ 第 48 轮：`_moe` 内省算子（预估/实测：每次 kernel 约 8 µs，少 8 个 kernel ≈ 3 ms/token）
_MOE_FAST = os.environ.get("FNX_MOE_FAST", "0") == "1"
_MOE_ROUTER_FAST = os.environ.get("FNX_MOE_ROUTER_FAST", "1" if _MOE_FAST else "0") == "1"
_MOE_ACT_BF16 = os.environ.get("FNX_MOE_ACT_BF16", "1" if _MOE_FAST else "0") == "1"
_SM_RAW = os.environ.get("FNX_SMALL_MERGE", "1")
_SM_MIX = _SM_RAW in ("1", "all", "mix")
_SM_SH = _SM_RAW in ("1", "all", "sh")
_SMALL_NAMES = ("a_down", "a_up", "m_down", "m_up", "a_inj", "m_inj",
                "sh_gate", "sh_up", "sh_down", "sh_gate2", "gate_w")
_ATTN_NAMES = ("w_qkv", "w_z", "w_b", "w_a", "w_out", "w_q", "w_k", "w_v", "w_o")



# ---------------- 基础算子 ----------------
# ★ 融合 RMSNorm：把原来 8–10 个 launch 换成 torch_npu 的单 kernel 融合算子。
# 实测（tools/check_rms_fused.py）：等价性 rel 0.152%（bf16 舍入级）、fp32 输入逐位一致；
# 单次调用 307.5 us -> 187.9 us（1.64x）。每层调用 2 次，dev1 的 22 层 eager 每 token
# 约省 5 ms。`FNX_FUSED_RMS=0` 可退回旧实现。
_FUSED_RMS = os.environ.get("FNX_FUSED_RMS", "1") == "1" and torch_npu is not None
_GAMMA_CACHE = {}
_ONES_CACHE = {}


def _gamma(w):
    """1+weight，按张量地址缓存（权重是常驻的，只算一次）。"""
    k = (w.data_ptr(), tuple(w.shape))
    g = _GAMMA_CACHE.get(k)
    if g is None:
        g = 1.0 + w.float()
        _GAMMA_CACHE[k] = g
    return g


def _ones(n, dev):
    k = (n, str(dev))
    t = _ONES_CACHE.get(k)
    if t is None:
        t = torch.ones(n, device=dev, dtype=torch.float32)
        _ONES_CACHE[k] = t
    return t


def rms_norm(x, w, gs=None):
    """对齐 Qwen4ExpTextRMSNorm：(1+weight)，fp32 内部计算。

    注意：`gs` 不为空时是**分组**归一化（每 gs 个元素一组），而 `w` 的长度等于 x 的
    最后一维（不是 gs）—— 权重在 flatten 之后按元素乘，所以 gamma 不能折进 w。
    """
    if _FUSED_RMS:
        try:
            # ★ 必须先转 fp32 再进融合算子：直接喂 bf16 会让 QSA 层的对拍误差从
            # 0.024% 退化到 1.857%（npu_rms_norm 对 bf16 输入的内部计算精度不够）。
            # 实测 fp32 输入下，融合算子与原实现**逐位一致**（rel 0.0000%）。
            xf = x.float()
            if gs is None:
                out, _ = torch_npu.npu_rms_norm(xf, _gamma(w), EPS)
                return out.type_as(x)
            xg = xf.view(*xf.shape[:-1], -1, gs)
            out, _ = torch_npu.npu_rms_norm(xg, _ones(gs, x.device), EPS)
            return (out.flatten(-2) * _gamma(w)).type_as(x)
        except Exception:  # noqa: BLE001
            pass
    xf = x.float()
    if gs is not None:
        xf = xf.view(*xf.shape[:-1], -1, gs)
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    if gs is not None:
        out = out.flatten(-2)
    out = out * (1.0 + w.float())
    return out.type_as(x)


def l2norm(x):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def inv_freq(device):
    idx = torch.arange(0, ROPE_DIM, 2, dtype=torch.float32, device=device)
    return 1.0 / (ROPE_THETA ** (idx / ROPE_DIM))


def rope_cs(pos, inv, dtype=torch.bfloat16):
    """pos: 0 维或 [B] int64 → (cos, sin) 形状 [ROPE_DIM] 或 [B, ROPE_DIM]。

    文本 mrope 退化为普通 RoPE；批处理时每个序列有自己的 pos。
    """
    f = pos.to(torch.float32)
    if f.dim() == 0:
        f = f * inv                     # [32]
    else:
        f = f.view(-1, 1) * inv         # [B,32]
    f = torch.cat([f, f], dim=-1)       # [64] 或 [B,64]
    return f.cos().to(dtype), f.sin().to(dtype)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)


def apply_rope(x, cos, sin, rot=ROPE_DIM):
    xr, xn = x[..., :rot], x[..., rot:]
    return torch.cat([xr * cos + rotate_half(xr) * sin, xn], dim=-1)


def deq(q, s, gs=None):
    if gs is None:
        gs = q.shape[-1] // s.shape[-1]
    sh = q.shape
    w = q.to(torch.bfloat16).view(*sh[:-1], sh[-1] // gs, gs)
    return (w * s.to(torch.bfloat16).unsqueeze(-1)).reshape(*sh)


# ---------------- 权重容器 ----------------
_fl = torch.nn.functional.linear


class QW:
    """稠密权重：**int8 + group scale 常驻 HBM**，每次调用才反量化。
    用于把稠密部分从 bf16 的 6.91 GiB 压到 3.46 GiB（48 层），代价 +2.6 ms/token。"""
    __slots__ = ("q", "s", "gs", "K", "chunk_n")

    def __init__(self, q, s, gs=None, K=None):
        self.q = q
        self.s = s
        self.gs = gs if gs is not None else q.shape[-1] // s.shape[-1]
        self.K = q.shape[-1] if K is None else K
        self.chunk_n = 0

    def deq_w(self):
        w = deq(self.q, self.s, self.gs)
        return w if self.K == w.shape[-1] else w[..., : self.K]

    def __call__(self, x):
        if self.chunk_n:
            n = self.q.shape[0]
            out = torch.empty(*x.shape[:-1], n, dtype=torch.bfloat16, device=x.device)
            for n0 in range(0, n, self.chunk_n):
                n1 = min(n0 + self.chunk_n, n)
                w = deq(self.q[n0:n1], self.s[n0:n1], self.gs)[..., : self.K]
                out[..., n0:n1] = _fl(x, w)
            return out
        return _fl(x, self.deq_w())


def lin(x, w):
    """w 可以是 bf16 Tensor、QW（按需反量化）、DenseW8（Cube 内融合反量化）或 Raw。"""
    if isinstance(w, QW):
        return w(x)
    if isinstance(w, DenseW8):
        return w(x)
    if isinstance(w, Raw):
        return _fl(x, w.w)
    return _fl(x, w)

class DenseW8:
    """稠密 int8 权重的 Cube 融合路径（aclnn `npu_weight_quant_batchmatmul`，W8 per-group）。

    第 38 轮实测（真实 L12，gs=128，M=1）：
      (a) 现状 `QW(x)`：每 token 把整块 int8 反量化成 bf16 再 linear —— **0.953 ms/层**
          （45 层 = 42.9 ms/token；profile 里就是那 24 ms 的 Mul+Cast 大头）
      (b) 装载期预反量化 + 纯 bf16 linear —— 0.158 ms/层，但要 **+55 MiB/层**（45 层 +2.5 GiB，装不下）
      (c) 本类：int8 常驻 + bf16 scale，反量化在 Cube 内 —— **0.263 ms/层**，**不占额外显存**
    数值与 (a) **逐位一致**（rel 0.00e+00：scale 是 fp16 直接转 bf16，int8 值精确）。

    scale 在装载期一次性转置成 `[G,N]` bf16（每个权重只有几十 KB），所以每 token 没有
    scale 拼接开销——这一点和专家（512 个专家、必须现场拼）根本不同。
    """
    __slots__ = ("w8t", "sc", "gs", "K", "N", "chunks")

    def __init__(self, q, s, K=None):
        gs = q.shape[-1] // s.shape[-1]
        kk = q.shape[-1] if K is None else K
        if kk % gs:                       # 有尾部 pad 且不整除组：不接这条路
            raise ValueError(f"DenseW8: K={kk} 不是 gs={gs} 的整数倍")
        g = kk // gs
        self.w8t = q[:, :kk].t().contiguous()                 # [K,N] int8
        self.sc = s[:, :g].t().contiguous().to(torch.bfloat16)  # [G,N] bf16
        self.gs = gs
        self.K = kk
        self.N = self.w8t.shape[1]
        self.chunks = None

    def make_chunks(self, chunk_n):
        """按 N 预切成连续块——算子要求 K、N ≤ 65535，lm_head 的 N=248320 必须切。

        实测意义：lm_head 现在是 dev1 上的 bf16 常驻（1.27 GB），一次 GEMM 要 5.7 ms；
        换 int8（0.64 GB）+ 4 段算子调用，读的字节腰斩，且走 Cube 内反量化。
        预切（而不是每次切片）是为了给算子连续内存：非连续视图它可能拒绝或隐式拷贝。
        """
        parts = []
        off = 0
        for w, s in zip(torch.split(self.w8t, chunk_n, dim=1),
                        torch.split(self.sc, chunk_n, dim=1)):
            n0 = off
            off += w.shape[1]
            parts.append((w.contiguous(), s.contiguous(), n0, off))
        self.chunks = parts
        self.w8t = None
        self.sc = None

    def deq_w(self):
        """一次性反量化（供 FNX_DEQ_SMALL 这类装载期预反量化用）。"""
        if self.chunks is not None:
            raise RuntimeError("DenseW8 已按 N 切块，不再支持一次性反量化")
        w = self.w8t.to(torch.bfloat16) * self.sc.repeat_interleave(self.gs, dim=0)
        return w.t().contiguous()                             # [N,K]

    def __call__(self, x):
        # 算子只吃 2 维 x（实测 3 维直接 161002: x's dim should be in range [2, 2]），
        # 而层里传进来的是 [b,1,H]；reshape 成 [b*1,H] 再还原，数值等价、无拷贝。
        sh = x.shape
        if x.dtype != torch.bfloat16:
            x = x.to(torch.bfloat16)
        x2 = x if x.dim() == 2 else x.reshape(-1, sh[-1])
        if self.chunks is None:
            y = torch_npu.npu_weight_quant_batchmatmul(
                x2, self.w8t, self.sc, antiquant_group_size=self.gs)
        else:
            y = torch.empty(x2.shape[0], self.N, dtype=torch.bfloat16, device=x2.device)
            for w, s, n0, n1 in self.chunks:
                y[:, n0:n1] = torch_npu.npu_weight_quant_batchmatmul(
                    x2, w, s, antiquant_group_size=self.gs)
        return y if x.dim() <= 2 else y.view(*sh[:-1], self.N)

    @staticmethod
    def merge(parts):
        """把**共用同一个输入**的几个投影沿 N 拼成一个权重，一次算子调用全出。

        实测动因（第 38 轮 profile）：r38 之后 `WeightQuantBatchmatmul` 变成最大单项
        16 ms/步，共 315 次调用 —— 平均每次只有 0.05 ms，**成本在发射不在算**。
        qkv 与 z 用的是同一个 h，拼成 [K, 10240+6144] 后一次调用即可（层内再切片，
        切片是对激活做，几十 KB，代价可忽略）。
        """
        assert parts and all(p is not None for p in parts)
        gs = parts[0].gs
        assert all(p.gs == gs and p.K == parts[0].K for p in parts), "组大小/K 不一致，不能拼"
        w = torch.cat([p.w8t for p in parts], dim=1)          # [K, ΣN]
        sc = torch.cat([p.sc for p in parts], dim=1)          # [G, ΣN]
        o = DenseW8.__new__(DenseW8)
        o.w8t, o.sc, o.gs, o.K, o.N = w, sc, gs, parts[0].K, w.shape[1]
        o.chunks = None
        return o


class Raw:
    """已反量化好的 bf16 权重 [N,K]。"""
    __slots__ = ("w",)

    def __init__(self, w):
        self.w = w


def pack4(q, chunk=64):
    """int8 容器（-8..7，1 值/字节）→ 真 4bit 打包，2 值/字节。无损。
    关键：绝不做步长 2 的切片（NPU 上退化成逐元素，实测 32 s/层）；
    改用 int32 视图把一个 int32 当成相邻两个 int16 处理，全程连续访存。
    落盘 dtype 用 int8 承载位型：先 to(uint8)（0..255 精确）再 view(int8)，不做数值截断。"""
    E, N, K = q.shape
    out = torch.empty(E, N, K // 2, dtype=torch.int8, device=q.device)
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        q32 = q[e0:e1].to(torch.int16).view(torch.int32)      # [c,N,K/2] 相邻两元素打包
        p = (q32 & 0x0F) | (((q32 >> 16) & 0x0F) << 4)        # 0..255
        out[e0:e1] = p.to(torch.int16).to(torch.uint8).view(torch.int8)
        del q32, p
    return out

class ExpQ:
    """专家权重：真 4bit 打包 int8 [E,N,K/2] + group scale [E,N,K/gs] 常驻，
    每 token 只 gather top-k 后解包 + 反量化。"""
    __slots__ = ("q", "s", "gs", "K", "N", "packed", "base", "qv2", "s4t",
                 "q_off", "dev", "v2ob")

    def __init__(self, q, s, gs, packed=True, base=None):
        self.s = s
        self.gs = gs
        self.K = q.shape[-1]
        self.N = q.shape[1]
        self.q = pack4(q) if packed else q
        self.packed = packed
        self.base = base
        self.qv2 = None
        self.s4t = None
        self.q_off = None                       # 宿主锁页副本（offload 模式下用）
        self.dev = self.s.device
        self.v2ob = os.environ.get("FNX_V2_OFFSET_BINARY") == "1"   # 运行时打包用二进制补码
        if os.environ.get("FNX_MOE_KERNEL") == "1":
            from .w4gemv_op import W4GemvOp
            # ★ 必须显式用 self.s 的设备：kernel 只认本卡指针。早先写 .npu()（=当前设备），
            # dev1 上的层会拿到 dev0 的地址 —— 这就是第 27 轮 §33–§35 追了半天的 507015 源头。
            _d = self.s.device
            # ★★ 真实 scale = s * base（g64 压缩时 s 存成 int8、每行 fp16 基准在 base 里）。
            # 早先直接拿 self.s 当 scale，等于把量级差 127 倍的数喂给 kernel -> 逐层放大到 inf/NaN。
            # base 必须在构造时传进来：load.py 原来是在构造**之后**才 eq.base = base。
            _sf = self.s.float()
            if self.base is not None:
                _sf = _sf * self.base.float().unsqueeze(-1)
            # ★★ kernel 里 `constexpr int32_t GS = 32`（scale 布局是 [K/32][N]），
            # 而线上 FNX_EXP_G=64 时 scale 是**每 64 个 k 一个**（K/64 行）。直接把 K/64 行
            # 传给 kernel，它会按 K/32 去读 -> 越界读 -> y 全 NaN（实测 NaN 241/359 个元素）。
            # 组内 64 个 k 共享一个 scale，所以沿 k 重复 2 次就是精确的 g32 视图，无精度损失。
            _gs_have = self.K // _sf.shape[-1]
            if _gs_have != 32:
                if _gs_have % 32:
                    raise RuntimeError(f"kernel 只支持 gs=32，当前 gs={_gs_have}，无法整除还原")
                _sf = _sf.repeat_interleave(_gs_have // 32, dim=-1)
            self.qv2 = torch.from_numpy(np.ascontiguousarray(
                W4GemvOp.to_v2_layout(self.q.cpu().numpy(), self.N, self.K))).to(_d)
            self.s4t = torch.from_numpy(np.ascontiguousarray(
                np.transpose(_sf.cpu().numpy(), (0, 2, 1)).astype(np.float16))).to(_d)
            assert self.s4t.shape[1] == self.K // 32, (self.s4t.shape, self.K)
            del _sf
            self.q = None
            # ★ kernel 模式下 `self.s` / `self.base` 已经没用了（s4t 是它的转置+融合版），
            # 但原来一直留着 → scale 在显存里存了两份（g32 fp16 约 0.15 GiB/层）。
            # 释放它，单卡装更多层才有可能。gather_deq 在 kernel 模式下本来就会 assert。
            self.s = None
            self.base = None
        elif os.environ.get("FNX_PACKED_TORCH") == "1":
            # ★ torch 路径也用打包 4bit 存储（1.17 GiB/层 vs int8 容器 2.34 GiB/层）。
            # 解包**必须走 bf16 算术**：实测 910B4 上 uint8>>4 要 7.3 ms、int16^8-8 要 18.2 ms，
            # 而 bf16 算术只要 0.03–0.09 ms/算子（见第 29 轮 §22）。
            # 并且**不能物化交织权重**（块状跨步写同样退化到 64 ms），
            # 所以解出 lo/hi 两个连续数组，下游做两次 GEMM（见 matmul()）。
            from .w4gemv_op import W4GemvOp
            # ★ q 可以传**宿主**张量：这样设备上只出现打包后的 qv2（1.17 GiB），
            # 不再需要 int8 容器（2.34 GiB）作为瞬态 —— 每层省 1.56 GiB 峰值，
            # 单卡可装层数从 37 提到 44（第 29 轮 §24）。
            _src = self.q if self.q.device.type == "cpu" else self.q.cpu()
            self.qv2 = torch.from_numpy(np.ascontiguousarray(
                W4GemvOp.to_v2_layout(_src.numpy(), self.N, self.K))).to(self.dev)
            self.q = None

    # ---------- 打包存储的解包 + 矩阵乘 ----------
    @classmethod
    def from_packed(cls, qv2, s8, base, dev):
        """直接从 LQP artifact 构建（不经过 int8 容器）—— 冷启动快、宿主峰值低。

        qv2 是 **offset-binary** v2 打包（n' = value+8），所以解包只需 `n' - 8`，
        比运行时打包（二进制补码，需要 remainder 还原符号）少 4 个算子。
        """
        o = cls.__new__(cls)
        o.q = None
        o.q_off = None
        o.s = s8.to(dev)
        o.base = base.to(dev)
        o.N = qv2.shape[1]
        o.K = qv2.shape[2] * 2
        # ★ gs 从 artifact 推断（不再写死 64）：g64 时 G=K/64，g128 时 G=K/128
        o.gs = o.K // s8.shape[-1]
        o.packed = True
        o.qv2 = qv2.to(dev)
        o.s4t = None
        o.dev = o.qv2.device
        o.v2ob = True
        # 自检：K 必须能被 128 整除（v2 的块单位）
        assert o.K % 128 == 0, o.K
        return o

    def _deq_v2(self, idx):
        """v2 打包 [E,N,K/2] -> (wlo, whi)，各是连续的 [k,N,K/2]，scale 已还原。

        第 34 轮按 msProf 的两处优化（都只减 kernel 数，不改数值语义）：
          1) `torch.div(pb, 16, rounding_mode="floor")` 代替 `floor(pb*0.0625)`
             —— 原来是"乘 + floor"两个 kernel，现在一个；
          2) offset-binary 的 `-8` 不再单独做一趟 16.4M 元素的减法，
             而是用 `addcmul(-8s, n', s)` 把 `(n'-8)*s` 合成**一个** kernel。
        """
        p = self.qv2.index_select(0, idx)
        k, N = p.shape[0], p.shape[1]
        S = self.K // 128
        pb = p.to(torch.bfloat16)                     # 0..255，bf16 尾数精确
        # ⚠️ 第 34 轮实测：`torch.div(pb, 16.0, rounding_mode="floor")` 比
        #   `torch.floor(pb * 0.0625)` **慢得多**（单个 /fnx/loss 从 30 s 拖到 >100 s 未完），
        #   所以虽然 msProf 里它是两个 kernel、这里是两个，实测仍是原来这条快。已回退。
        hi = torch.floor(pb * 0.0625)                 # floor(byte/16)
        lo = pb - hi * 16.0
        fold8 = _FOLD8 and self.v2ob and not _S8_SIM128 and not _SIM3
        if not self.v2ob:
            # 二进制补码：值 = (n ^ 8) - 8 = ((n+8) % 16) - 8
            lo = torch.remainder(lo + 8.0, 16.0) - 8.0
            hi = torch.remainder(hi + 8.0, 16.0) - 8.0
        elif not fold8:
            # offset-binary：值 = n' - 8
            lo = lo - 8.0
            hi = hi - 8.0
        if _SIM3:
            # ★★ 3-bit 专家可行性实验（第 32 轮追加）：
            #   4-bit 每组 16 个电平、步长 s；3-bit 在同动态范围下只有 8 个电平、步长 2s。
            #   所以把 q 量化到偶数即可**精确模拟 3-bit 的量化误差**（scale 不变）。
            #   只改数值、不改磁盘布局，用来量"专家降到 3-bit"的质量代价，
            #   对应能省 300 MiB/层 × 45 ≈ 13 GiB —— 足够把 48 层全捕获 + 全部注意力预反量化。
            lo = torch.round(lo * 0.5) * 2.0
            hi = torch.round(hi * 0.5) * 2.0
        se = self.s.index_select(0, idx).to(torch.bfloat16)      # [k,N,G]
        if self.base is not None:
            se = se * self.base.index_select(0, idx).to(torch.bfloat16).unsqueeze(-1)
        if _S8_SIM128:
            # ★ 实验用：把相邻两个 g64 scale 合并成一个（等效 g128）。
            #   注意必须在**乘 base 之后**做算平均（此时已经是真实 scale，浮点），
            #   千万不要再 round —— 早先手滑加了 round_()，等于把权重取整，
            #   mean_nll 直接炸到 8.12（这本身也说明 q*scale 的精度对结果极其敏感）。
            se = se.mean(dim=-1, keepdim=True).expand_as(se).contiguous()
        G = se.shape[-1]
        if G == S:
            # ★ g128 布局：每 128 个 k 只有 1 个 scale（第 32 轮，省 18.8 MiB/层）
            sv = se.view(k, N, S, 1)
            s_lo = s_hi = sv
        else:
            # g64 布局：每 128 个 k 两个 scale（v2 块的 lo/hi 各一个）
            sv = se.view(k, N, S, 2)
            s_lo, s_hi = sv[..., 0:1], sv[..., 1:2]
        lo64 = lo.view(k, N, S, 64)
        hi64 = hi.view(k, N, S, 64)
        if fold8:
            # 优化 2：w = (n' - 8) * s = n'*s + (-8s)，一次 addcmul 完成
            wlo = torch.addcmul(-8.0 * s_lo, lo64, s_lo).reshape(k, N, S * 64)
            whi = torch.addcmul(-8.0 * s_hi, hi64, s_hi).reshape(k, N, S * 64)
        else:
            wlo = (lo64 * s_lo).reshape(k, N, S * 64)
            whi = (hi64 * s_hi).reshape(k, N, S * 64)
        return wlo, whi

    def matmul(self, xb, idx):
        """xb: [k,1,K] bf16 -> [k,N]。打包存储走两次 GEMM，否则走原来的一次。"""
        if self.qv2 is not None:
            return _gemm2(xb, *self._deq_v2(idx), self.K)
        w = self.gather_deq(idx)
        return torch.bmm(xb, w.transpose(1, 2)).squeeze(1)

    def to_host_offload(self):
        """★ 把专家权重从显存搬到**宿主锁页**，设备侧不再常驻 [E,N,K]。

        实测依据（第 29 轮 §20）：每专家一个连续 blob 的分段 H2D 能跑 **21.4 GB/s**
        （10×2.34 MB = 1.15 ms），而宿主侧 `index_select` 之后再拷只有 0.74 GB/s。
        所以这里坚持"按专家逐块拷"，绝不在宿主做 gather。

        省下 2.34 GiB/层（int8 容器）。代价：这层的 MoE 变成 eager（H2D 源地址依赖路由结果，
        不能被图捕获），除非把 GEMV 段单独捕获（后续工作）。
        """
        if self.q is None:
            return False
        self.q_off = self.q.to("cpu").pin_memory()
        self.q = None
        return True

    def gather_deq(self, idx):
        assert self.q is not None or self.q_off is not None, "kernel 路径下 self.q 已释放"
        s = self.s.index_select(0, idx)
        if self.base is not None:
            s = s.to(torch.bfloat16) * self.base.index_select(0, idx).to(torch.bfloat16).unsqueeze(-1)
        if self.q_off is not None:
            # offload 分支：逐专家连续 H2D 到共享暂存区（21.4 GB/s 的那个模式）
            ids = idx.detach().to("cpu").tolist()          # 一次宿主同步：H2D 源地址依赖路由结果
            qq = _staging(ids, self.N, self.K, self.dev)
            for j, e in enumerate(ids):
                qq[j].copy_(self.q_off[e], non_blocking=True)
            k, N = qq.shape[0], qq.shape[1]
            w = qq.to(torch.bfloat16).view(k, N, self.K // self.gs, self.gs)
            return (w * s.to(torch.bfloat16).unsqueeze(-1)).reshape(k, N, self.K)
        if not self.packed:
            qq = self.q.index_select(0, idx)
            k, N = qq.shape[0], qq.shape[1]
            w = qq.to(torch.bfloat16).view(k, N, self.K // self.gs, self.gs)
            return (w * s.to(torch.bfloat16).unsqueeze(-1)).reshape(k, N, self.K)
        p = self.q.index_select(0, idx)
        k, N = p.shape[0], p.shape[1]
        p16 = p.to(torch.int16) & 0xFF
        lo = ((p16 & 0x0F) ^ 8) - 8
        hi = (((p16 >> 4) & 0x0F) ^ 8) - 8
        q = torch.stack([lo, hi], dim=-1).reshape(k, N, self.K).to(torch.bfloat16)
        w = q.view(k, N, self.K // self.gs, self.gs)
        return (w * s.to(torch.bfloat16).unsqueeze(-1)).reshape(k, N, self.K)


class ExpW4:
    """专家权重：aclnn per-group W4A16（`npu_weight_quant_batchmatmul`）专用容器。

    为什么值得：msProf 显示 88.5 ms/step 的设备时间里 46 ms（56%）是 4-bit 解包的
    **Vector 逐元素链**（Mul 25.8 + Cast 9.2 + Subs 5.0 + Sub 4.3 + Floor 1.9），
    而 HBM 只跑到约 95 GB/s（等效 2% 带宽利用率）—— 卡在 Vector 吞吐，不在带宽。

    关键实测（第 37 轮，自包含探针 backend/remote/p32{2,5,6}_*.py）：
      * CANN 9.1 的 aclnnWeightQuantBatchMatmulV2 **支持 Atlas A2（910B4）per-group**，
        weight 用 `npu_convert_weight_to_int4pack` 的 int32 容器，scale 形状 (K/gs, N)，
        **必须与 x 同为 bf16**（传 fp16 报 161002 “x's dtype must be same as antiquantScale's”）。
        历史结论“只吃 per-channel”是把这个 dtype 不匹配当成了不支持。
      * 数值：与 bf16 反量化+bmm 的参考 rel 8.8e-3（bf16 舍入级）。
      * 图捕获：replay vs eager rel 0.000e+00，换专家 ids 后仍一致。
      * 时序（E=512，真实分布）：gu 拼接一次调用 0.043 ms（16.4 MB 权重），
        dn 对角法 0.054 ms；加权重/scales 拼接合计 ≈ 0.32 ms/层。

    存储口径（offline repack，见 tools/repack_w4a16.py）：
      w4  int32 [K,E,N/8]（转置：index_select(dim=1) 后即 [K,k,N/8] 连续，省一次 permute）
      s8t int8  [G,E,N]  （= 原 s8 的转置，逐位不变）
      base bf16 [E,N]    （原 fp16 基准转 bf16，只为喂算子）
    """
    __slots__ = ("w4", "s8t", "base", "K", "N", "E", "G", "gs", "_dg", "_dgf", "expert_major")

    def __init__(self, w4, s8t, base, gs, expert_major=False):
        self.w4 = w4              # [K,E,N/8] int32
        self.s8t = s8t            # [G,E,N] int8
        self.base = base          # [E,N] bf16
        self.expert_major = expert_major
        self.K, self.E = (w4.shape[1], w4.shape[0]) if expert_major else (w4.shape[0], w4.shape[1])
        self.N = w4.shape[2] * 8
        self.G = s8t.shape[0]
        self.gs = gs
        # 对角下标常驻（图捕获期不要新建 arange，避免多一次分配）
        dev = w4.device
        k = TOPK
        self._dg = torch.arange(k, dtype=torch.int64, device=dev)
        # ★ 第 48 轮：把 [k,k,N] 的对角取法压成**一次** index_select。
        #   原写法 `y.view(k,k,N)[dg,dg]` 是两次高级索引（约 2–3 个 kernel），
        #   在"每次 kernel 约 8 µs"的图里就是白花的 16–24 µs/层。
        _dk = torch.arange(k, dtype=torch.int64, device=dev)
        self._dgf = (_dk * (k * self.N) + _dk * self.N).unsqueeze(1) \
            + torch.arange(self.N, dtype=torch.int64, device=dev).unsqueeze(0)

    def _wcat(self, ids):
        if self.expert_major:
            p = self.w4.index_select(0, ids).permute(1, 0, 2).contiguous()
            return p.reshape(self.K, -1)
        p = torch.index_select(self.w4, 1, ids)                 # [K,k,N/8] 连续
        return p.reshape(self.K, p.shape[1] * (self.N // 8))

    def _scat(self, ids):
        sc = torch.index_select(self.s8t, 1, ids)               # [G,k,N] int8
        bs = torch.index_select(self.base, 0, ids).unsqueeze(0)  # [1,k,N] bf16
        s = sc.to(torch.bfloat16)
        if _MOE_FAST:
            s.mul_(bs)                       # 原地乘：少一次分配（数值不变）
        else:
            s = s * bs
        return s.reshape(self.G, -1)

    def apply(self, x, ids):
        """x [M,K] bf16（M=1 或 k），ids [k] -> [M, k*N]。"""
        return torch_npu.npu_weight_quant_batchmatmul(
            x, self._wcat(ids), self._scat(ids), antiquant_group_size=self.gs)

    def apply_diag(self, x, ids):
        """k 行各用 ids 里对应的专家：一次调用算 k×k 块，只取对角 -> [k,N]。"""
        k = x.shape[0]
        assert k == TOPK, k
        y = self.apply(x, ids)                                  # [k, k*N]
        if _MOE_FAST:
            # 一次 index_select 取到对角（原写法是两次高级索引）
            return y.view(-1).index_select(0, self._dgf.reshape(-1)).view(k, self.N)
        return y.view(k, k, self.N)[self._dg, self._dg]

    def apply_grouped(self, x, group_ends, host_ends=None):
        """Rows sorted by expert, one per-group int4 GMM, no Cartesian blocks."""
        if not self.expert_major:
            raise RuntimeError("grouped prefill requires the expert-major artifact")
        if host_ends is not None:
            # Bound CANN's temporary dequantization workspace. This eager
            # prefill path reads routing metadata once for both projections.
            size = int(os.environ.get("FNX_MOE_GROUP_SPLIT", "32"))
            outputs = []
            for first in range(0, self.E, size):
                last = min(first + size, self.E)
                begin = host_ends[first - 1] if first else 0
                end = host_ends[last - 1]
                if end == begin:
                    continue
                scales = self.s8t[:, first:last].permute(1, 0, 2).to(torch.bfloat16) * self.base[first:last, None, :]
                # Slice-based int4 GMM returned wrong results in this build.
                # Materialize bounded weights/inputs at storage offset zero.
                weight = self.w4[first:last].clone()
                inputs = x[begin:end].clone()
                outputs.append(torch_npu.npu_grouped_matmul(
                    [inputs], [weight], antiquant_scale=[scales],
                    group_list=group_ends[first:last] - begin, split_item=3,
                    group_type=0, output_dtype=torch.bfloat16)[0])
            return torch.cat(outputs, dim=0)
        scales = self.s8t.permute(1, 0, 2).to(torch.bfloat16) * self.base[:, None, :]
        return torch_npu.npu_grouped_matmul(
            [x], [self.w4], antiquant_scale=[scales], group_list=group_ends,
            split_item=3, group_type=0, output_dtype=torch.bfloat16)[0]


_STG = {}


def _split_halves(x, K):
    """把 k 维按 v2 的分块切成 lo/hi 两半（[128s,128s+64) 与 [128s+64,128s+128)）。

    切片作用在**激活**上（[10,1,K]，几十 KB），代价可忽略；
    绝不作用在权重上 —— 那是当初 FNX_PACK=1 慢死的原因。
    """
    S = K // 128
    v = x.reshape(x.shape[0], x.shape[1], S, 2, 64)     # 用 reshape：调用方可能传 expanded 张量
    a = v[..., 0, :].reshape(x.shape[0], x.shape[1], S * 64).contiguous()
    b = v[..., 1, :].reshape(x.shape[0], x.shape[1], S * 64).contiguous()
    return a, b


def _gemm2(xb, wlo, whi, K):
    """y = xb·W^T，其中 W 由两半拼成（不物化交织权重）。"""
    xlo, xhi = _split_halves(xb, K)
    return (torch.bmm(xlo, wlo.transpose(1, 2)).squeeze(1)
            + torch.bmm(xhi, whi.transpose(1, 2)).squeeze(1))


def _staging(ids, N, K, dev):
    """共享的设备暂存区 [k, N, K] int8 —— 所有 offload 层复用同一块（层是串行执行的）。

    共享而不是每层一块：22 层各一块要 1.08 GiB 设备内存，共享只要 49 MB。
    """
    k = len(ids)
    key = (k, N, K, str(dev))
    t = _STG.get(key)
    if t is None:
        t = torch.empty(k, N, K, dtype=torch.int8, device=dev)
        _STG[key] = t
    return t


class KVStore:
    def __init__(self, P, device, n):
        self.P = P
        self.position_major = os.environ.get("FNX_KV_POSITION_MAJOR", "0") == "1"
        def allocate():
            if self.position_major:
                return torch.zeros(B, P, KVH, HD, dtype=torch.bfloat16, device=device).permute(0, 2, 1, 3)
            return torch.zeros(B, KVH, P, HD, dtype=torch.bfloat16, device=device)
        self.k = [allocate() for _ in range(n)]
        self.v = [allocate() for _ in range(n)]


def gated_residual(h, hc_norm_w, w_down, w_up, w_inj, n_inj=0):
    """n_inj > 0 表示 w_down 已把 block_inject 拼在尾部（最后 n_inj 列是 inj）。

    第 39 轮：`w_down`（→LOWRANK=320）与 `w_inj`（→HC=4）都吃同一个 hn，拼成一次 GEMM
    可以少一次发射（Captured 路径里 MatMul 单次 ~13 µs，但其中大半是固定开销）。
    拼接后数值不变：每列的点积与原来完全相同。
    """
    b = h.shape[0]
    hn = rms_norm(h, hc_norm_w, gs=H)
    di = lin(hn, w_down)
    if n_inj:
        dp, ip = di[..., :-n_inj], di[..., -n_inj:]
    else:
        dp, ip = di, None
    mix = F.silu(dp / HC)
    mix = torch.sigmoid(lin(mix, w_up))
    mixed = (mix.view(b, 1, HC, H) * hn.view(b, 1, HC, H)).mean(dim=-2).view(b, 1, H)
    if n_inj:
        return mixed, (2.0 * torch.sigmoid(ip / HC)).view(b, 1, HC)
    if w_inj is None:
        return mixed
    inj = 2.0 * torch.sigmoid(lin(hn, w_inj) / HC)
    return mixed, inj.view(b, 1, HC)


def inject(hyper, inj, y):
    """hyper + y·inj（广播到 HCH）。每层调用 2 次，原来要 3 个 kernel（乘/reshape/加）。

    第 41 轮：`mul → reshape → add` 换成一次 `addcmul`（数值等价：同一次乘加，
    只是没有中间的 reshape 物化）。张量只有 10240 个元素，比当年那个 16.4M 元素的
    dequant 场景小三个数量级，所以第 34 轮"addcmul 更慢"的结论在这里不适用——需实测。
    """
    b = hyper.shape[0]
    if _INJECT_CM:
        # addcmul 要求三者能广播到同一形状：把 hyper 也看成 [b,1,HC,H]
        out = torch.addcmul(hyper.view(b, 1, HC, H), y.view(b, 1, 1, H), inj.unsqueeze(-1))
        return out.view(b, 1, HCH)
    return hyper + (y.view(b, 1, 1, H) * inj.unsqueeze(-1)).reshape(b, 1, HCH)


def delta_chunk_fp32(q, k, v, log_g, beta, state):
    """Parallel delta-rule chunk using an fp32 triangular solve.

    q already contains the decode path's rounded 1/sqrt(D) scaling.
    Preserve the fp32 recurrent state rather than casting it to bf16.
    """
    T = q.shape[0]
    if T > 64 and os.environ.get("FNX_DELTA_BATCH", "0") == "1":
        return delta_chunk_batched_fp32(q, k, v, log_g, beta, state)
    if T > 64:
        outputs = []
        for begin in range(0, T, 64):
            end = min(begin + 64, T)
            out, state = delta_chunk_fp32(q[begin:end], k[begin:end], v[begin:end],
                                          log_g[begin:end], beta[begin:end], state)
            outputs.append(out)
        return torch.cat(outputs, dim=0), state
    q, k, v = (x.transpose(0, 1).float() for x in (q, k, v))
    beta = beta.reshape(T, NVH).transpose(0, 1).float()
    cumul = log_g.transpose(0, 1).cumsum(-1)
    upper = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
    decay = (cumul.unsqueeze(-1) - cumul.unsqueeze(-2)).masked_fill(upper, float('-inf')).exp()
    kb = k * beta.unsqueeze(-1)
    vb = v * beta.unsqueeze(-1)
    lower = (torch.bmm(kb, k.transpose(1, 2)) * decay).tril(-1)
    system = lower + torch.eye(T, dtype=torch.float32, device=q.device)
    rhs = torch.cat((vb, kb * cumul.exp().unsqueeze(-1)), dim=-1)
    solver = os.environ.get("FNX_DELTA_SOLVER", "triangular")
    if solver == "blockmatmul":
        # Invert 8x8 diagonal blocks, then solve the block-lower system.
        # Both finite series have at most eight stages of dependency.
        nb = (T + 7) // 8
        width = nb * 8
        lp = F.pad(lower, (0, width - T, 0, width - T))
        rp = F.pad(rhs, (0, 0, 0, width - T))
        diagonal = lp.view(NVH, nb, 8, nb, 8).permute(0, 1, 3, 2, 4).reshape(NVH, nb * nb, 8, 8)
        diagonal = diagonal.index_select(1, torch.arange(nb, device=q.device) * (nb + 1)).reshape(NVH * nb, 8, 8)
        inv = torch.eye(8, device=q.device).expand(NVH * nb, 8, 8)
        power = -diagonal
        for stage in range(3):
            inv = inv + torch.bmm(power, inv)
            if stage < 2:
                power = torch.bmm(power, power)
        solved = torch.bmm(inv, rp.reshape(NVH * nb, 8, HVD + HKD)).reshape(NVH, width, HVD + HKD)
        transformed = torch.bmm(inv, lp.reshape(NVH * nb, 8, width)).reshape(NVH, width, width)
        block_ids = torch.arange(width, device=q.device) // 8
        across = block_ids[:, None] > block_ids[None, :]
        power = -transformed.masked_fill(~across, 0)
        for stage in range((nb - 1).bit_length()):
            solved = solved + torch.bmm(power, solved)
            if stage + 1 < (nb - 1).bit_length():
                power = torch.bmm(power, power)
        solved = solved[:, :T]
    elif solver == "matmul":
        # Forward substitution in 8-row blocks. A global Neumann product
        # is unstable for strongly correlated real keys (large intermediate
        # powers cancel). Small local systems bound that amplification.
        pieces = []
        for begin in range(0, T, 8):
            end = min(begin + 8, T)
            part = rhs[:, begin:end]
            if begin:
                part = part - torch.bmm(lower[:, begin:end, :begin], torch.cat(pieces, dim=1))
            power = -lower[:, begin:end, begin:end]
            stages = (end - begin - 1).bit_length()
            for stage in range(stages):
                part = part + torch.bmm(power, part)
                if stage + 1 < stages:
                    power = torch.bmm(power, power)
            pieces.append(part)
        solved = torch.cat(pieces, dim=1)
    else:
        solved = torch.linalg.solve_triangular(system, rhs, upper=False, unitriangular=True)
    u, w = solved[..., :HVD], solved[..., HVD:]
    updates = u - torch.bmm(w, state[0])
    intra = torch.bmm(q, k.transpose(1, 2)) * decay
    out = torch.bmm(q * cumul.exp().unsqueeze(-1), state[0]) + torch.bmm(intra, updates)
    end_keys = k * (cumul[:, -1:] - cumul).exp().unsqueeze(-1)
    final = state[0] * cumul[:, -1:].exp().unsqueeze(-1) + torch.bmm(end_keys.transpose(1, 2), updates)
    return out.transpose(0, 1).to(torch.bfloat16), final.unsqueeze(0)


def delta_chunk_batched_fp32(q, k, v, log_g, beta, state):
    """Prepare independent 64-token systems together; keep state recurrence fp32.

    The systems depend on keys/decay, not on the incoming recurrent state.
    Only the state update and output need a sequential loop over blocks.
    """
    T = q.shape[0]
    blocks = (T + 63) // 64
    length = blocks * 64
    def rows(x, dim):
        x = F.pad(x, (0, 0, 0, 0, 0, length - T))
        return x.view(blocks, 64, NVH, dim).permute(0, 2, 1, 3).reshape(-1, 64, dim).float()
    qa, ka, va = (rows(x, d) for x, d in ((q, HKD), (k, HKD), (v, HVD)))
    ba = F.pad(beta.reshape(T, NVH), (0, 0, 0, length - T))
    ba = ba.view(blocks, 64, NVH).transpose(1, 2).reshape(-1, 64).float()
    ga = F.pad(log_g, (0, 0, 0, length - T))
    cumul = ga.view(blocks, 64, NVH).transpose(1, 2).reshape(-1, 64).cumsum(-1)
    upper = torch.ones(64, 64, dtype=torch.bool, device=q.device).triu(1)
    decay = (cumul.unsqueeze(-1) - cumul.unsqueeze(-2)).masked_fill(upper, float('-inf')).exp()
    kb, vb = ka * ba.unsqueeze(-1), va * ba.unsqueeze(-1)
    lower = (torch.bmm(kb, ka.transpose(1, 2)) * decay).tril(-1)
    rhs = torch.cat((vb, kb * cumul.exp().unsqueeze(-1)), -1)
    heads = blocks * NVH
    diagonal = lower.view(heads, 8, 8, 8, 8).permute(0, 1, 3, 2, 4).reshape(heads, 64, 8, 8)
    diagonal = diagonal.index_select(1, torch.arange(8, device=q.device) * 9).reshape(heads * 8, 8, 8)
    inv = torch.eye(8, device=q.device).expand(heads * 8, 8, 8)
    power = -diagonal
    for stage in range(3):
        inv = inv + torch.bmm(power, inv)
        if stage < 2:
            power = torch.bmm(power, power)
    solved = torch.bmm(inv, rhs.reshape(heads * 8, 8, HVD + HKD)).reshape(heads, 64, HVD + HKD)
    transformed = torch.bmm(inv, lower.reshape(heads * 8, 8, 64)).reshape(heads, 64, 64)
    bid = torch.arange(64, device=q.device) // 8
    power = -transformed.masked_fill(~(bid[:, None] > bid[None, :]), 0)
    for stage in range(3):
        solved = solved + torch.bmm(power, solved)
        if stage < 2:
            power = torch.bmm(power, power)
    u = solved[..., :HVD].reshape(blocks, NVH, 64, HVD)
    w = solved[..., HVD:].reshape(blocks, NVH, 64, HKD)
    intra = (torch.bmm(qa, ka.transpose(1, 2)) * decay).reshape(blocks, NVH, 64, 64)
    qe = (qa * cumul.exp().unsqueeze(-1)).reshape(blocks, NVH, 64, HKD)
    end_keys = (ka * (cumul[:, -1:] - cumul).exp().unsqueeze(-1)).reshape(blocks, NVH, 64, HKD)
    end_decay = cumul[:, -1:].exp().reshape(blocks, NVH, 1, 1)
    current = state[0]
    outputs = []
    for block in range(blocks):
        updates = u[block] - torch.bmm(w[block], current)
        out = torch.bmm(qe[block], current) + torch.bmm(intra[block], updates)
        current = current * end_decay[block] + torch.bmm(end_keys[block].transpose(1, 2), updates)
        outputs.append(out.transpose(0, 1))
    return torch.cat(outputs, 0)[:T].to(torch.bfloat16), current.unsqueeze(0)


class FnLayer:
    def __init__(self, idx, ltype, W, device):
        self.idx = idx
        self.ltype = ltype
        self.dev = device
        self.kv = None
        self.ki = 0
        self.inv = inv_freq(device)
        self.use_ple = ltype == "ple"
        self.index_enabled = os.environ.get("FNX_INDEXER", "0") == "1" and ltype == "full"
        self.index_k = None
        self.index_keys = None
        self.short_qsa = False
        self._dn_lens = {}
        if self.index_enabled:
            if B != 1:
                raise ValueError("QSA indexer currently requires FNX_BATCH=1")
            self.index_w = W["self_attn.indexer.index_qk_proj.weight"]
            self.index_qn = W["self_attn.indexer.q_layernorm.weight"]
            self.index_kn = W["self_attn.indexer.k_layernorm.weight"]

        if ltype in ("linear", "ple"):
            self.S = torch.zeros(B, NVH, HKD, HVD, dtype=torch.float32, device=device)
        else:
            self.kv = None
        self.conv = torch.zeros(B, CONV_DIM, 3, dtype=torch.bfloat16, device=device)
        if self.use_ple:
            self.ple_conv = torch.zeros(B, HCH, PLE_CONV_LEN, dtype=torch.bfloat16, device=device)

        self.a_hc_norm = W["attn_hyper_connection.hc_norm.weight"]
        self.a_down = W["attn_hyper_connection.input_mix_weight_down.weight"]
        self.a_up = W["attn_hyper_connection.input_mix_weight_up.weight"]
        self.a_inj = W["attn_hyper_connection.block_inject_weight.weight"]
        self.m_hc_norm = W["mlp_hyper_connection.hc_norm.weight"]
        self.m_down = W["mlp_hyper_connection.input_mix_weight_down.weight"]
        self.m_up = W["mlp_hyper_connection.input_mix_weight_up.weight"]
        self.m_inj = W["mlp_hyper_connection.block_inject_weight.weight"]

        if ltype in ("linear", "ple"):
            p = "linear_attn."
            self.w_qkv = W[p + "in_proj_qkv.weight"]
            self.w_z = W[p + "in_proj_z.weight"]
            self.w_b = W[p + "in_proj_b.weight"]
            self.w_a = W[p + "in_proj_a.weight"]
            self.w_out = W[p + "out_proj.weight"]
            self.conv_w = W[p + "conv1d.weight"].view(CONV_DIM, 4)
            self.A_log = W[p + "A_log"].float()
            self.dt_bias = W[p + "dt_bias"].float()
            self.dn_norm_w = W[p + "norm.weight"].float()
            # rms_norm() 是 (1+w) 语义，这里需要直接乘，预先减 1
            self.dn_norm_g = (self.dn_norm_w - 1.0).contiguous()
        if ltype in ("full",):
            p = "self_attn."
            self.w_q = W[p + "q_proj.weight"]
            self.w_k = W[p + "k_proj.weight"]
            self.w_v = W[p + "v_proj.weight"]
            self.w_o = W[p + "o_proj.weight"]
            self.qn_w = W[p + "q_norm.weight"]
            self.kn_w = W[p + "k_norm.weight"]

        if self.use_ple:
            p = "ple."
            self.pl_key = W[p + "key_proj.weight"]
            self.pl_val = W[p + "value_proj.weight"]
            self.pl_nk = W[p + "norm_key.weight"]
            self.pl_nq = W[p + "norm_query.weight"]
            self.pl_nc = W[p + "norm_conv.weight"]
            self.pl_conv_w = W[p + "conv1d.weight"].view(HCH, 4)

        p = "mlp."
        self.gate_w = W[p + "gate.weight"]
        self.exp_gu = W[p + "experts.gate_up_proj"]
        self.exp_dn = W[p + "experts.down_proj"]
        self.sh_gate = W[p + "shared_expert.gate_proj.weight"]
        self.sh_up = W[p + "shared_expert.up_proj.weight"]
        self.sh_down = W[p + "shared_expert.down_proj.weight"]
        self.sh_gate2 = W[p + "shared_expert_gate.weight"]

        # ★ 装载期预反量化：把这些小权重一次性变成 bf16 常驻（数值逐位相同）
        if _DEQ_SMALL:
            n = 0
            for nm in _SMALL_NAMES:
                Wx = getattr(self, nm, None)
                if Wx is not None and hasattr(Wx, "deq_w"):
                    setattr(self, nm, Raw(Wx.deq_w()))
                    n += 1
            self.n_predeq = n
        else:
            self.n_predeq = 0

        # ★ 注意力投影预反量化（按层数预算；dev1 的层永远做，显存富余）
        on_dev1 = str(device).endswith("1")
        if _DEQ_ATTN > 0 and (on_dev1 or idx < _DEQ_ATTN):
            n = 0
            for nm in _ATTN_NAMES:
                Wx = getattr(self, nm, None)
                if Wx is not None and hasattr(Wx, "deq_w"):
                    setattr(self, nm, Raw(Wx.deq_w()))
                    n += 1
            self.n_predeq_attn = n
        else:
            self.n_predeq_attn = 0

        # ★★ 第 38 轮：把共用同一输入的稠密投影拼成一次算子调用（省发射，约 −3 ms/token）
        #   拼完必须把原件置 None：否则同一份权重在显存里放两遍（+55 MiB/层）。
        self.w_qkvz = None
        self.w_ab = None
        self.w_qkvf = None
        self.qkvz_ba = False
        self.n_cat = 0
        if _DENSE_CAT:
            # ★ 第 42 轮：linear 层的 4 个投影（qkv/z/b/a）都吃同一个 h，理论上能一次调用出完。
            #   **实测不赚**：65.1/65.7 vs 基线 64.7/65.6（噪声内，nll 不变 2.1735）。
            #   原因：w_b/w_a 只有 48 列（0.25 MB），那一次调用的固定开销远小于 49 µs 的平均值；
            #   而合并后的大调用 N 从 10240 涨到 16480，多读的字节把省下的抵消了。
            #   保留开关 `FNX_DENSE_CAT4=1` 以便将来 op 版变化时复测。
            four = ("w_qkv", "w_z", "w_b", "w_a")
            ws4 = [getattr(self, n, None) for n in four]
            if _CAT4 and all(isinstance(w, DenseW8) for w in ws4):
                self.w_qkvz = DenseW8.merge(ws4)
                self.qkvz_ba = True
                for n in four:
                    setattr(self, n, None)
            else:
                names = ("w_qkv", "w_z")
                ws = [getattr(self, n, None) for n in names]
                if all(isinstance(w, DenseW8) for w in ws):
                    self.w_qkvz = DenseW8.merge(ws)
                    for n in names:
                        setattr(self, n, None)
                names = ("w_b", "w_a")
                ws = [getattr(self, n, None) for n in names]
                if all(isinstance(w, DenseW8) for w in ws):
                    self.w_ab = DenseW8.merge(ws)
                    for n in names:
                        setattr(self, n, None)
            names = ("w_q", "w_k", "w_v")
            ws = [getattr(self, n, None) for n in names]
            if all(isinstance(w, DenseW8) for w in ws):
                self.w_qkvf = DenseW8.merge(ws)
                for n in names:
                    setattr(self, n, None)
            if self.w_qkvz is not None or self.w_ab is not None:
                self.n_cat = int(self.w_qkvz is not None) + int(self.w_ab is not None)

        # ★★ 第 39 轮：bf16 小权重里"共用同一输入"的也拼成一次 GEMM（MatMul 单次 13 µs）
        #   mixer：w_down(→320) 与 w_inj(→HC=4) 都吃 hn；shared expert：sh_gate 与 sh_up 都吃 x2。
        self.a_di = self.m_di = self.sh_gu = None
        self.sh_gw = False
        if _SM_MIX:
            for pre in ("a", "m"):
                wd, wi = getattr(self, pre + "_down", None), getattr(self, pre + "_inj", None)
                if isinstance(wd, Raw) and isinstance(wi, Raw):
                    setattr(self, pre + "_di", Raw(torch.cat([wd.w, wi.w], dim=0)))
                    setattr(self, pre + "_down", None)
                    setattr(self, pre + "_inj", None)
        if _SM_SH:
            # 第 40 轮：把"同样吃 x2"的 4 个权重一次算完再切：
            #   sh_gate(640) + sh_up(640) + sh_gate2(1) + gate_w(512)  →  一次 GEMM（N=1793）
            # 动机同上：MatMul 单次 ~13 µs 里绝大部分是固定开销。
            # 注意：N 变了会改变框架 tiling → 累加顺序变，/fnx/loss 可能微动（实测见第 40 轮记录）。
            names = ("sh_gate", "sh_up", "sh_gate2", "gate_w")
            ws = [getattr(self, n, None) for n in names]
            if isinstance(ws[0], Raw) and isinstance(ws[1], Raw):
                ws = [ws[0], ws[1]] + [w if isinstance(w, Raw) else None for w in ws[2:]]
                if all(w is not None for w in ws):
                    self.sh_gu = Raw(torch.cat([w.w for w in ws], dim=0))
                    for n in names:
                        setattr(self, n, None)
                    self.sh_gw = True
                else:
                    self.sh_gu = Raw(torch.cat([ws[0].w, ws[1].w], dim=0))
                    self.sh_gate = None
                    self.sh_up = None
                    self.sh_gw = False

    # ---------- DeltaNet 单步 ----------
    def _deltanet(self, h):
        b = h.shape[0]
        if self.w_qkvz is not None:
            qz = lin(h, self.w_qkvz)[:, 0]                 # [b, CONV+VAL(+2*NVH)]，切片是视图
            mq = qz[:, :CONV_DIM]
            if self.qkvz_ba:
                # qkv+z+b+a 四合一（第 42 轮）
                z = torch.sigmoid(qz[:, CONV_DIM:CONV_DIM + VAL_DIM].float()).view(b, NVH, HVD)
                beta = torch.sigmoid(
                    qz[:, CONV_DIM + VAL_DIM:CONV_DIM + VAL_DIM + NVH]).view(b, NVH, 1)
                a = qz[:, CONV_DIM + VAL_DIM + NVH:]
            else:
                z = torch.sigmoid(qz[:, CONV_DIM:].float()).view(b, NVH, HVD)
                beta = a = None
        else:
            mq = lin(h, self.w_qkv).view(b, CONV_DIM)
            z = torch.sigmoid(lin(h, self.w_z).view(b, NVH, HVD).float())
            beta = a = None
        buf = torch.cat([self.conv, mq.unsqueeze(-1)], dim=-1)              # [b,D,4]
        y = (buf * self.conv_w.unsqueeze(0)).sum(-1)
        y = F.silu(y)
        self.conv.copy_(buf[..., 1:])
        q, k, v = torch.split(y, [KEY_DIM, KEY_DIM, VAL_DIM], dim=-1)
        if beta is None:
            if self.w_ab is not None:
                ba = lin(h, self.w_ab)[:, 0]                   # [b, NVH+NVH]
                beta = torch.sigmoid(ba[:, :NVH]).view(b, NVH, 1)
                a = ba[:, NVH:]
            else:
                beta = torch.sigmoid(lin(h, self.w_b)).view(b, NVH, 1)
                a = lin(h, self.w_a)
        g = (-torch.exp(self.A_log) * F.softplus(a.float() + self.dt_bias)).view(b, NVH)

        rep = NVH // NKH
        # 第 40 轮试过：把 S 看成 [b,NKH,rep,HKD,HVD] 直接广播，免掉 repeat_interleave。
        # 实测 **反而慢 0.7 ms**（5 维张量的逐元素效率不如物化两次 + 4 维），已回退。
        q = l2norm(q.view(b, NKH, HKD)).repeat_interleave(rep, dim=1) / math.sqrt(HKD)
        k = l2norm(k.view(b, NKH, HKD)).repeat_interleave(rep, dim=1)
        v = v.view(b, NVH, HVD)

        S = self.S * torch.exp(g)[..., None, None]                          # 先衰减
        if os.environ.get("FNX_DN_DECODE_BMM", "0") == "1":
            kv_mem = torch.bmm(k.float().reshape(-1, 1, HKD), S.reshape(-1, HKD, HVD)).reshape(b, NVH, HVD)
            delta = (v - kv_mem) * beta
            S = torch.addcmul(S, k.float().unsqueeze(-1), delta.unsqueeze(-2))
            o = torch.bmm(q.float().reshape(-1, 1, HKD), S.reshape(-1, HKD, HVD)).reshape(b, NVH, HVD)
        else:
            kv_mem = (S * k.unsqueeze(-1)).sum(dim=-2)
            delta = (v - kv_mem) * beta
            S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
            o = (S * q.unsqueeze(-1)).sum(dim=-2)
        self.S.copy_(S)

        o = o.to(torch.bfloat16)
        if _DN_RMS:
            # ★ 第 42 轮：这里原来是手写的 4 乘 + 2 规约（pow/mean/rsqrt/mul/mul），
            #   换成引擎里已有的融合 `npu_rms_norm`（内部 fp32 计算）。
            #   ⚠️ rms_norm() 内部用 (1+weight) 语义，而这里要的是**直接乘** dn_norm_w，
            #   所以装载期预存了 dn_norm_g = dn_norm_w - 1。第一版漏了这一步，nll 直接炸到 2.52。
            xf = rms_norm(o, self.dn_norm_g).float() * z
        else:
            xf = o.float()
            xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
            xf = self.dn_norm_w * xf.to(torch.bfloat16)
            xf = xf * z                             # output_gate_type = "sigmoid"（不是 silu）
        return lin(xf.to(torch.bfloat16).view(b, VAL_DIM), self.w_out)

    def _indexer_mask(self, h, pos):
        """QSA block selection matching the vendored reference (B=1).

        Four query heads score mean-pooled blocks of four raw keys. Keep the
        best 512 complete blocks plus the incomplete causal tail. All indices
        and positions stay on device so this also works in NPUGraph replay.
        """
        T = h.shape[0]
        P = self.kv.P
        qk = lin(h, self.index_w).reshape(T, 640)
        q = rms_norm(qk[:, :512].reshape(T, 4, 128), self.index_qn)
        self.index_k.index_copy_(0, pos, qk[:, 512:].contiguous())
        cos, sin = rope_cs(pos, self.inv)
        q = apply_rope(q, cos.unsqueeze(1), sin.unsqueeze(1))
        groups = P // 4
        starts = torch.arange(groups, device=self.dev, dtype=torch.int64) * 4
        pooled = self.index_k[:groups * 4].view(groups, 4, 128).float().mean(1).to(h.dtype)
        pooled = rms_norm(pooled, self.index_kn)
        cos, sin = rope_cs(starts, self.inv)
        keys = apply_rope(pooled, cos, sin)
        scores = torch.matmul(q.float(), keys.float().transpose(0, 1)).relu().sum(1) / math.sqrt(128)
        complete = starts[None, :] + 4 <= pos[:, None] + 1
        scores = scores.masked_fill(~complete, NEG)
        chosen = scores.topk(min(512, groups), dim=-1).indices
        blocks = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, chosen, True) & complete
        # repeat_interleave on bool lowers very poorly on 910B4 (81 ms for
        # [64,2048]); broadcast each block's four tokens in one materialization.
        selected = blocks.unsqueeze(-1).expand(T, groups, 4).reshape(T, groups * 4)
        if P % 4:
            selected = F.pad(selected, (0, P % 4))
        columns = torch.arange(P, device=self.dev)[None, :]
        tail_start = ((pos[:, None] + 1) // 4) * 4
        tail = (columns >= tail_start) & (columns <= pos[:, None])
        return selected | tail

    def _indexer_indices(self, h, pos):
        """Return at most 2048 selected tokens plus three causal tail tokens.

        Score query tiles to bound workspace independently of the context
        window; never construct a [T,P] token mask or replicate full KV heads.
        """
        T, P = h.shape[0], self.kv.P
        qk = lin(h, self.index_w).reshape(T, 640)
        q = rms_norm(qk[:, :512].reshape(T, 4, 128), self.index_qn)
        self._record_index_keys(qk[:, 512:], pos)
        cos, sin = rope_cs(pos, self.inv)
        q = apply_rope(q, cos.unsqueeze(1), sin.unsqueeze(1))
        groups = P // 4
        starts = torch.arange(groups, device=self.dev, dtype=torch.int64) * 4
        if os.environ.get("FNX_INDEX_CACHE", "0") == "1":
            keys = self.index_keys.transpose(0, 1)
        else:
            pooled = self.index_k[:groups * 4].view(groups, 4, 128).float().mean(1).to(h.dtype)
            keys = rms_norm(pooled, self.index_kn)
            cos, sin = rope_cs(starts, self.inv)
            keys = apply_rope(keys, cos, sin).float().transpose(0, 1)
        selected, validity = [], []
        tile = int(os.environ.get("FNX_INDEX_TILE", "32"))
        for begin in range(0, T, tile):
            end = min(begin + tile, T)
            p = pos[begin:end]
            scores = torch.matmul(q[begin:end].float().reshape(-1, 128), keys)
            scores = scores.view(end - begin, 4, groups).relu().sum(1) / math.sqrt(128)
            complete = starts[None, :] + 4 <= p[:, None] + 1
            chosen = scores.masked_fill(~complete, NEG).topk(min(512, groups), -1).indices
            tokens = (chosen[:, :, None] * 4 + torch.arange(4, device=self.dev)).reshape(end - begin, -1)
            valid = tokens < ((p[:, None] + 1) // 4) * 4
            tail = ((p[:, None] + 1) // 4) * 4 + torch.arange(3, device=self.dev)
            tokens = torch.cat((tokens, tail), -1)
            valid = torch.cat((valid, tail <= p[:, None]), -1)
            # Physical order reproduces dense attention's accumulation order.
            ordered = torch.where(valid, tokens, P).float().sort(-1).values.to(torch.int64)
            selected.append(ordered.clamp_max(P - 1))
            validity.append(ordered < P)
        return torch.cat(selected), torch.cat(validity)

    def _record_index_keys(self, raw, pos):
        self.index_k.index_copy_(0, pos, raw.contiguous())
        if os.environ.get("FNX_INDEX_CACHE", "0") != "1":
            return
        groups = self.kv.P // 4
        if self.index_keys is None:
            pooled = self.index_k[:groups * 4].view(groups, 4, 128).float().mean(1).to(raw.dtype)
            starts = torch.arange(groups, device=self.dev) * 4
            cos, sin = rope_cs(starts, self.inv)
            self.index_keys = apply_rope(rms_norm(pooled, self.index_kn), cos, sin).float()
        # Duplicate group ids write identical values. Incomplete groups are
        # excluded by the causal mask; recompute them when more raw keys arrive.
        group_ids = pos // 4
        rows = (group_ids[:, None] * 4 + torch.arange(4, device=self.dev)).reshape(-1)
        pooled = self.index_k.index_select(0, rows).view(-1, 4, 128).float().mean(1).to(raw.dtype)
        cos, sin = rope_cs(group_ids * 4, self.inv)
        keys = apply_rope(rms_norm(pooled, self.index_kn), cos, sin).float()
        self.index_keys.index_copy_(0, group_ids, keys)

    def _sparse_qsa_attention(self, h, q, pos, kc, vc):
        ids, valid = self._indexer_indices(h, pos)
        outputs = []
        T, width = ids.shape
        tile = int(os.environ.get("FNX_QSA_TILE", "32"))
        fused = os.environ.get("FNX_SPARSE_FLASH", "0") == "1"
        for begin in range(0, T, tile):
            end = min(begin + tile, T)
            n = end - begin
            rows = ids[begin:end].reshape(-1)
            if self.kv.position_major:
                keys = kc[0].transpose(0, 1).index_select(0, rows).view(n, width, KVH, HD)
                values = vc[0].transpose(0, 1).index_select(0, rows).view(n, width, KVH, HD)
                if not fused:
                    keys = keys.permute(0, 2, 1, 3).reshape(n * KVH, width, HD)
                    values = values.permute(0, 2, 1, 3).reshape(n * KVH, width, HD)
            else:
                keys = kc[0].index_select(1, rows).view(KVH, n, width, HD).permute(1, 0, 2, 3).reshape(n * KVH, width, HD)
                values = vc[0].index_select(1, rows).view(KVH, n, width, HD).permute(1, 0, 2, 3).reshape(n * KVH, width, HD)
            if fused:
                if self.kv.position_major:
                    queries = q[begin:end].reshape(n, 1, NAH, HD).contiguous()
                    layout = "BSND"
                else:
                    queries = q[begin:end].reshape(n, NAH, 1, HD).contiguous()
                    keys, values = keys.view(n, KVH, width, HD), values.view(n, KVH, width, HD)
                    layout = "BNSD"
                out = torch_npu.npu_fused_infer_attention_score(
                    queries, keys, values, num_heads=NAH, num_key_value_heads=KVH,
                    input_layout=layout, scale=HD ** -0.5,
                    atten_mask=(~valid[begin:end])[:, None, None, :])[0]
                outputs.append(out.reshape(n, NAH, HD))
                continue
            queries = q[begin:end].reshape(n * KVH, NAH // KVH, HD)
            scores = torch.bmm(queries, keys.transpose(1, 2)) * (HD ** -0.5)
            mask = valid[begin:end, None, None, :].expand(n, KVH, 1, width).reshape(n * KVH, 1, width)
            weights = torch.softmax(scores.masked_fill(~mask, NEG).float(), -1).to(torch.bfloat16)
            out = torch.bmm(weights, values).reshape(n, NAH, HD)
            outputs.append(out)
        return torch.cat(outputs).reshape(T, NAH * HD)

    # ---------- QSA ----------
    def _qsa(self, h, pos):
        b = h.shape[0]
        cos, sin = rope_cs(pos, self.inv)
        if b > 1:
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)     # [b,1,64] 便于广播
        if self.w_qkvf is not None:
            qkv = lin(h, self.w_qkvf)[:, 0]                     # [b, 6144+512+512]
            nq = NAH * 2 * HD
            qg = qkv[:, :nq].reshape(b, NAH, 2 * HD)
            k = rms_norm(qkv[:, nq:nq + KVH * HD].reshape(b, KVH, HD), self.kn_w)
            v = qkv[:, nq + KVH * HD:].reshape(b, KVH, HD)
        else:
            qg = lin(h, self.w_q).view(b, NAH, 2 * HD)
            k = rms_norm(lin(h, self.w_k).view(b, KVH, HD), self.kn_w)
            v = lin(h, self.w_v).view(b, KVH, HD)
        q = rms_norm(qg[..., :HD], self.qn_w)
        gate = qg[..., HD:]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        P = min(self.kv.P, 2048) if self.short_qsa else self.kv.P
        # 逐行写各自的 pos（one-hot 形式，图捕获已验证可用；比 index_put 稳）
        kc, vc = self.kv.k[self.ki][:, :, :P], self.kv.v[self.ki][:, :, :P]
        sparse = os.environ.get("FNX_QSA_SPARSE", "0") == "1" and b == 1
        if sparse:
            if self.kv.position_major:
                kc[0].transpose(0, 1).index_copy_(0, pos, k.contiguous())
                vc[0].transpose(0, 1).index_copy_(0, pos, v.contiguous())
            else:
                kc[0].index_copy_(1, pos, k.transpose(0, 1).contiguous())
                vc[0].index_copy_(1, pos, v.transpose(0, 1).contiguous())
        else:
            onehot = F.one_hot(pos, P).to(torch.bfloat16).view(b, 1, P, 1)
            kc.copy_(kc * (1.0 - onehot) + k.view(b, KVH, 1, HD) * onehot)
            vc.copy_(vc * (1.0 - onehot) + v.view(b, KVH, 1, HD) * onehot)
        if sparse and self.index_enabled and not self.short_qsa:
            o = self._sparse_qsa_attention(h, q, pos, kc, vc)
            return lin(o * torch.sigmoid(gate.reshape(b, NAH * HD)), self.w_o)

        mask = (torch.arange(P, device=self.dev)[None, :] <= pos.view(b, 1)) \
            .view(b, 1, 1, P)
        if self.index_enabled:
            if self.short_qsa:
                # All visible tokens are selected before the budget boundary.
                # Still record raw keys, so a later long request can reuse state.
                qk = lin(h, self.index_w).reshape(b, 640)
                self._record_index_keys(qk[:, 512:], pos)
            else:
                mask = mask & self._indexer_mask(h, pos).view(b, 1, 1, P)
        Kk = kc.repeat_interleave(NAH // KVH, dim=1)                        # [b,NAH,P,HD]
        Vv = vc.repeat_interleave(NAH // KVH, dim=1)
        att = torch.matmul(q.view(b, NAH, 1, HD), Kk.transpose(2, 3)) * (HD ** -0.5)
        att = att.masked_fill(~mask, NEG)
        att = torch.softmax(att.float(), dim=-1).to(torch.bfloat16)
        o = torch.matmul(att, Vv).view(b, NAH * HD)
        return lin(o * torch.sigmoid(gate.reshape(b, NAH * HD)), self.w_o)

    # ---------- 分块（预填）路径：第 46 轮 ----------
    # 动机：prefill 现在是逐 token 跑同一套 forward，每 token 都要把整层权重读一遍。
    # 实测摊薄系数（第 45 轮，真实 L00 权重）：稠密 GEMM T=32 时 32.7×，MoE 权重 31.1×。
    # 分块只批量化"与时间无关"的部分（投影/GEMM），顺序依赖的递推仍逐 token。
    def _deltanet_chunk(self, h, T):
        """h: [T,1,H]（同一个序列连续的 T 个位置）。返回 [T,1,H]。"""
        if self.w_qkvz is not None:
            qz = lin(h, self.w_qkvz)[:, 0]                      # [T, CONV+VAL(+2NVH)]
            mq = qz[:, :CONV_DIM]
            if self.qkvz_ba:
                z = torch.sigmoid(qz[:, CONV_DIM:CONV_DIM + VAL_DIM].float()).view(T, NVH, HVD)
                beta = torch.sigmoid(
                    qz[:, CONV_DIM + VAL_DIM:CONV_DIM + VAL_DIM + NVH]).view(T, NVH, 1)
                a = qz[:, CONV_DIM + VAL_DIM + NVH:]
            else:
                z = torch.sigmoid(qz[:, CONV_DIM:].float()).view(T, NVH, HVD)
                beta = a = None
        else:
            mq = lin(h, self.w_qkv).view(T, CONV_DIM)
            z = torch.sigmoid(lin(h, self.w_z).view(T, NVH, HVD).float())
            beta = a = None
        if beta is None:
            if self.w_ab is not None:
                ba = lin(h, self.w_ab)[:, 0]
                beta = torch.sigmoid(ba[:, :NVH]).view(T, NVH, 1)
                a = ba[:, NVH:]
            else:
                beta = torch.sigmoid(lin(h, self.w_b)).view(T, NVH, 1)
                a = lin(h, self.w_a)
        log_g = (-torch.exp(self.A_log) * F.softplus(a.float() + self.dt_bias)).view(T, NVH)
        g = torch.exp(log_g)

        # 卷积：把 3 列历史拼在左边，再按 4 个位移求和（与逐 token 的滑窗等价）
        # ★ 必须用 stack + sum(-1) 而不是 4 次逐项相加：逐 token 路径是 `(buf*w).sum(-1)`，
        #   sum 用成对归约、手写加法是左结合，bf16 下差 1 ulp（实测 y max 7.8e-3），
        #   再经 MoE topk 边界放大成整句 logits 偏 6.9e-2。改成同一种归约后逐位相等。
        seq = torch.cat([self.conv[0], mq.transpose(0, 1)], dim=-1)      # [D, T+3]
        w = self.conv_w                                                   # [D,4]
        y = torch.stack([w[:, j:j + 1] * seq[:, j:j + T] for j in range(4)],
                        dim=-1).sum(-1)                                   # [D,T]
        y = F.silu(y).transpose(0, 1)                                     # [T,D]
        self.conv.copy_(seq[:, T:T + 3].unsqueeze(0))
        q, k, v = torch.split(y, [KEY_DIM, KEY_DIM, VAL_DIM], dim=-1)

        rep = NVH // NKH
        qa = l2norm(q.view(T, NKH, HKD)).repeat_interleave(rep, dim=1) / math.sqrt(HKD)
        ka = l2norm(k.view(T, NKH, HKD)).repeat_interleave(rep, dim=1)
        va = v.view(T, NVH, HVD)

        # 递推（顺序依赖，只能逐 token；这里每 token 只有 6 个算子，且没有 GEMM）
        dn_mode = os.environ.get("FNX_PREFILL_DN", "torch")
        if dn_mode == "fp32_chunk":
            o, state = delta_chunk_fp32(qa, ka, va, log_g, beta, self.S)
            self.S.copy_(state)
        elif dn_mode == "native":
            if T not in self._dn_lens:
                self._dn_lens[T] = torch.tensor([T], dtype=torch.int32, device=self.dev)
            o, state = torch_npu.npu_chunk_gated_delta_rule(
                qa.contiguous(), ka.contiguous(), va.contiguous(),
                beta=beta.reshape(T, NVH).to(torch.bfloat16), g=log_g.contiguous(),
                initial_state=self.S.transpose(-1, -2).to(torch.bfloat16).contiguous(),
                actual_seq_lengths=self._dn_lens[T], scale=1.0)
            self.S.copy_(state.transpose(-1, -2))
        else:
            S = self.S[0]
            outs = []
            for t in range(T):
                St = S * g[t][:, None, None]
                kv_mem = (St * ka[t].unsqueeze(-1)).sum(dim=-2)
                delta = (va[t] - kv_mem) * beta[t]
                St = St + ka[t].unsqueeze(-1) * delta.unsqueeze(-2)
                outs.append((St * qa[t].unsqueeze(-1)).sum(dim=-2))
                S = St
            self.S[0].copy_(S)
            o = torch.stack(outs, dim=0).to(torch.bfloat16)
        if _DN_RMS:
            xf = rms_norm(o, self.dn_norm_g).float() * z
        else:
            xf = o.float()
            xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
            xf = self.dn_norm_w * xf.to(torch.bfloat16)
            xf = xf * z
        return lin(xf.to(torch.bfloat16).view(T, VAL_DIM), self.w_out)

    def _qsa_chunk(self, h, pos0, T):
        """h: [T,1,H]；pos0 是这 T 个 token 的起始位置。"""
        dynamic_pos = torch.is_tensor(pos0)
        if dynamic_pos:
            # Device position stays live across graph replays; no .item() in capture.
            pos = pos0.reshape(-1)[:1] + torch.arange(T, dtype=torch.int64, device=self.dev)
        else:
            pos = torch.arange(pos0, pos0 + T, dtype=torch.int64, device=self.dev)
        cos, sin = rope_cs(pos, self.inv)                    # [T, ROPE_DIM]
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)         # [T,1,64] 广播到 head
        if self.w_qkvf is not None:
            qkv = lin(h, self.w_qkvf)[:, 0]
            nq = NAH * 2 * HD
            qg = qkv[:, :nq].reshape(T, NAH, 2 * HD)
            k = rms_norm(qkv[:, nq:nq + KVH * HD].reshape(T, KVH, HD), self.kn_w)
            v = qkv[:, nq + KVH * HD:].reshape(T, KVH, HD)
        else:
            qg = lin(h, self.w_q).view(T, NAH, 2 * HD)
            k = rms_norm(lin(h, self.w_k).view(T, KVH, HD), self.kn_w)
            v = lin(h, self.w_v).view(T, KVH, HD)
        q = rms_norm(qg[..., :HD], self.qn_w)
        gate = qg[..., HD:]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        P = min(self.kv.P, 2048) if self.short_qsa else self.kv.P
        fixed_short = getattr(self, "reference_short", False) and self.short_qsa
        PT = P if dynamic_pos or fixed_short else pos0 + T
        kc, vc = self.kv.k[self.ki][:, :, :P], self.kv.v[self.ki][:, :, :P]
        # 逐行写 pos（分块时一次写 T 行；eager 下 index_copy_ 可用，不必用 one-hot 广播）
        if self.kv.position_major:
            kc[0].transpose(0, 1).index_copy_(0, pos, k.contiguous())
            vc[0].transpose(0, 1).index_copy_(0, pos, v.contiguous())
        else:
            kc[0].index_copy_(1, pos, k.transpose(0, 1).contiguous())
            vc[0].index_copy_(1, pos, v.transpose(0, 1).contiguous())
        if self.index_enabled and not self.short_qsa and os.environ.get("FNX_QSA_SPARSE", "0") == "1":
            o = self._sparse_qsa_attention(h, q, pos, kc, vc)
            return lin(o * torch.sigmoid(gate.reshape(T, NAH * HD)), self.w_o)

        grouped_qsa = os.environ.get("FNX_QSA_GQA", "0") == "1"
        if not grouped_qsa:
            Kr = kc[0].repeat_interleave(NAH // KVH, dim=0).unsqueeze(0)
            Vr = vc[0].repeat_interleave(NAH // KVH, dim=0).unsqueeze(0)
        qq = q.transpose(0, 1).unsqueeze(0)                               # [1,NAH,T,HD]
        # ★ 只算前 PT 列：pos0+T 之后的 KV 本来就要被 mask 掉，全算 P=2048 列在 prefill
        #   早期是纯粹的浪费（pos0=0,T=8 时白算 256 倍）。切片开销为 0（视图）。
        if not grouped_qsa:
            att = torch.matmul(qq, Kr[:, :, :PT].transpose(2, 3)) * (HD ** -0.5)
        qpos = pos.view(1, 1, T, 1)
        kmask = (torch.arange(PT, device=self.dev)[None, None, None, :] <= qpos)
        if self.index_enabled:
            if self.short_qsa:
                qk = lin(h, self.index_w).reshape(T, 640)
                self._record_index_keys(qk[:, 512:], pos)
            else:
                kmask = kmask & self._indexer_mask(h, pos)[None, None, :, :PT]
        if grouped_qsa:
            outputs = []
            for begin in range(0, T, 64):
                end = min(begin + 64, T)
                n = end - begin
                queries = q[begin:end].transpose(0, 1).reshape(KVH, (NAH // KVH) * n, HD)
                scores = torch.bmm(queries, kc[0, :, :PT].transpose(1, 2)).reshape(1, NAH, n, PT) * (HD ** -0.5)
                scores = scores.masked_fill(~kmask[:, :, begin:end], NEG)
                weights = torch.softmax(scores.float(), dim=-1).to(torch.bfloat16)
                out = torch.bmm(weights.reshape(KVH, (NAH // KVH) * n, PT), vc[0, :, :PT])
                outputs.append(out.reshape(NAH, n, HD).transpose(0, 1))
            o = torch.cat(outputs, dim=0).reshape(T, NAH * HD)
        else:
            att = att.masked_fill(~kmask, NEG)
            att = torch.softmax(att.float(), dim=-1).to(torch.bfloat16)
            o = torch.matmul(att, Vr[:, :, :PT]).squeeze(0)
            o = o.transpose(0, 1).reshape(T, NAH * HD)
        return lin(o * torch.sigmoid(gate.reshape(T, NAH * HD)), self.w_o)

    def forward_chunk(self, h, pos0, T):
        """分块前向：h [T,1,H]。MoE 暂按 token 循环（第 ② 步再做并集/分组）。"""
        if self.use_ple and not _NO_PLE:
            raise RuntimeError("分块 prefill 暂不支持 PLE（生产是 FNX_NO_PLE=1）")
        if self.a_di is not None:
            mixed, inj = gated_residual(h, self.a_hc_norm, self.a_di, self.a_up, None, n_inj=HC)
        else:
            mixed, inj = gated_residual(h, self.a_hc_norm, self.a_down, self.a_up, self.a_inj)
        if self.ltype in ("linear", "ple"):
            y = self._deltanet_chunk(mixed, T)
        else:
            y = self._qsa_chunk(mixed, pos0, T)
        h = inject(h, inj, y)
        if self.m_di is not None:
            mixed, inj = gated_residual(h, self.m_hc_norm, self.m_di, self.m_up, None, n_inj=HC)
        else:
            mixed, inj = gated_residual(h, self.m_hc_norm, self.m_down, self.m_up, self.m_inj)
        if os.environ.get("FNX_MOE_GROUP", "0") == "1" and T > 1:
            y = self._moe(mixed)
        else:
            y = torch.cat([self._moe(mixed[t:t + 1]) for t in range(T)], dim=0)
        return inject(h, inj, y)

    # ---------- PLE ----------
    def _ple(self, h, ngram_emb):
        kn = rms_norm(lin(ngram_emb, self.pl_key), self.pl_nk, gs=H).view(1, 1, HC, H)
        val = lin(ngram_emb, self.pl_val)
        qn = rms_norm(h, self.pl_nq, gs=H).view(1, 1, HC, H)
        gate = (kn * qn).sum(dim=-1, keepdim=True) / math.sqrt(H)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gv = torch.sigmoid(gate) * val.unsqueeze(-2)                        # [1,1,4,2560]
        gvn = rms_norm(gv.reshape(1, 1, HCH), self.pl_nc, gs=H)
        gv = gv.reshape(1, 1, HCH)
        buf = torch.cat([self.ple_conv, gvn.transpose(1, 2)], dim=-1)       # [1,D,10]
        y = (buf[..., 0::3] * self.pl_conv_w.unsqueeze(0)).sum(-1)
        self.ple_conv.copy_(buf[..., 1:])
        return gv + F.silu(y).unsqueeze(1)

    # ---------- MoE ----------
    def _moe(self, x):
        b = x.shape[0]
        x2 = x.view(b, H)
        if self.sh_gu is not None:
            gu = lin(x2, self.sh_gu)                         # [b, 2*MI (+1 gate2) (+V)]
            sh = lin(F.silu(gu[:, :MI]) * gu[:, MI:2 * MI], self.sh_down)
            if self.sh_gw:
                # gate2(1 列) 与 gate_w(V 列) 也在这块里：一次 GEMM 全出
                sh = torch.sigmoid(gu[:, 2 * MI:2 * MI + 1]) * sh
                logits = gu[:, 2 * MI + 1:]
            else:
                sh = torch.sigmoid(lin(x2, self.sh_gate2)) * sh
                logits = lin(x2, self.gate_w)
        else:
            sh = lin(F.silu(lin(x2, self.sh_gate)) * lin(x2, self.sh_up), self.sh_down)
            sh = torch.sigmoid(lin(x2, self.sh_gate2)) * sh
            logits = lin(x2, self.gate_w)
        if _MOE_ROUTER_FAST:
            # ★ 第 48 轮：softmax 的分母在 `w/w.sum()` 里会被约掉，所以可以
            #   ① 直接在 **原始 logits** 上 topk（少一次 fp32 cast + 512 路 softmax）
            #   ② 只对这 10 个值做 softmax（等价于"topk 后再归一化"）
            #   算子数 6 -> 3，数学上与原写法相同（bf16 末位可能差 1 ulp）。
            w10, idx = torch.topk(logits, TOPK, dim=-1)           # [b,10]
            w = torch.softmax(w10.float(), dim=-1).to(torch.bfloat16)
        else:
            probs = torch.softmax(logits.float(), dim=-1)
            w, idx = torch.topk(probs, TOPK, dim=-1)                 # [b,10]
            w = (w / w.sum(dim=-1, keepdim=True)).to(torch.bfloat16)

        if isinstance(self.exp_gu, ExpW4):
            if b > 1 and os.environ.get("FNX_MOE_GROUP", "0") == "1":
                ids = idx.reshape(-1)
                # Expert ids are 0..511 and exactly representable in fp32.
                # int64 ArgSort falls back to AI CPU on this CANN build.
                order = torch.argsort(ids.float())
                counts = torch.zeros(self.exp_gu.E, dtype=torch.int64, device=x.device)
                counts.scatter_add_(0, ids, torch.ones_like(ids))
                ends = counts.cumsum(0)
                host_ends = ends.cpu().tolist() if int(os.environ.get("FNX_MOE_GROUP_SPLIT", "0")) else None
                rows = x2.index_select(0, order // TOPK).to(torch.bfloat16)
                gu = self.exp_gu.apply_grouped(rows, ends, host_ends)
                act = (F.silu(gu[:, :MI].float()) * gu[:, MI:].float()).to(torch.bfloat16)
                yd = self.exp_dn.apply_grouped(act, ends, host_ends)
                restored = torch.empty_like(yd).index_copy_(0, order, yd)
                y = (restored.view(b, TOPK, H) * w.unsqueeze(-1)).sum(1)
                return (y + sh).view(b, 1, H)
            # ★★ 第 37 轮：aclnn per-group W4A16（解包/乘 scale/GEMM 全在 AI Core 内）
            #   实测（真实形状，E=512 分布，图内 replay）0.32 ms/层 vs torch 解包路径 1.0 ms。
            #   gu：10 个专家权重沿 N 拼成一次调用（单序列只有一行 x，天然共用）；
            #   dn：10 行各用不同专家 -> 同样一次调用，取 [k,k,N] 的对角块
            #       （Cube 富余，多算的 10x 只要 0.011 ms，比 10 次单专家调用 0.467 ms 便宜 8 倍）。
            if b != 1:
                raise RuntimeError("FNX_MOE_W4A16 只支持 FNX_BATCH=1（第 37 轮）")
            y = self.exp_gu.apply(x2.to(torch.bfloat16), idx[0]).view(TOPK, 2 * MI)
            if _MOE_ACT_BF16:
                # ★ 第 48 轮：silu*mul 全程留 bf16，省掉 2 次升 fp32 + 1 次回 bf16（3 个 kernel）
                act = F.silu(y[:, :MI]) * y[:, MI:]
            else:
                act = (F.silu(y[:, :MI].float()) * y[:, MI:].float()).to(torch.bfloat16)
            y2 = self.exp_dn.apply_diag(act, idx[0])
            y2 = (y2 * w[0].unsqueeze(-1)).sum(dim=0, keepdim=True)
            return (y2 + sh).view(1, 1, H)

        if b > 1:
            # ★ 批量：把 b×TOPK 个 (token,专家) 行摊平成一次 gather + 两次 GEMM。
            #   专家的权重量与 b 无关地被摊薄（同一批里重复选中的专家会被重复读一次，
            #   可接受；将来可按出现频次去重）。
            ids = idx.reshape(-1)                                # [b*10]
            rows = x2.to(torch.bfloat16).unsqueeze(1).expand(b, TOPK, H) \
                .reshape(b * TOPK, 1, H)
            wgu = self.exp_gu.matmul(rows, ids)                  # [b*10, N]
            act = (F.silu(wgu[:, :MI].float()) * wgu[:, MI:].float()).to(torch.bfloat16)
            y = self.exp_dn.matmul(act.unsqueeze(1), ids)         # [b*10, N2]
            y = (y.view(b, TOPK, -1) * w.unsqueeze(-1)).sum(dim=1)   # [b, N2]
            return (y + sh).view(b, 1, H)

        if os.environ.get("FNX_MOE_KERNEL") == "1":
            _op = _get_moe_op(x2.device)
            if _MOE_DBG:
                print(f"[moe-dbg L{self.idx}] gu: x2={tuple(x2.shape)} H={H} K={self.exp_gu.K} N={self.exp_gu.N} ids={idx[0].tolist()}", flush=True)
            _y = _op.run(self.exp_gu.qv2, self.exp_gu.s4t,
                         x2.to(torch.float32).expand(TOPK, H).contiguous(),
                         TOPK, self.exp_gu.N, self.exp_gu.K,
                         expert_ids=idx[0].to(torch.int32), lay=2)
            _act = (F.silu(_y[:, :MI].float()) * _y[:, MI:].float()).contiguous()
            if _MOE_DBG:
                print(f"[moe-dbg L{self.idx}] gu done y={tuple(_y.shape)} act={tuple(_act.shape)}", flush=True)
            _y2 = _op.run(self.exp_dn.qv2, self.exp_dn.s4t, _act,
                          TOPK, self.exp_dn.N, self.exp_dn.K,
                          expert_ids=idx[0].to(torch.int32), lay=2)
            if _MOE_DBG:
                print(f"[moe-dbg L{self.idx}] dn done y2={tuple(_y2.shape)}", flush=True)
            _s = _op.weighted_sum(_y2, w[0])
            if _MOE_DBG:
                print(f"[moe-dbg L{self.idx}] ok", flush=True)
            return (_s.view(1, -1) + sh).view(1, 1, H)
        wgu = self.exp_gu.matmul(x2.to(torch.bfloat16).expand(TOPK, H).unsqueeze(1), idx[0])
        act = (F.silu(wgu[:, :MI].float()) * wgu[:, MI:].float()).to(torch.bfloat16)
        y = self.exp_dn.matmul(act.unsqueeze(1), idx[0])                    # [10,N]
        y = (y * w[0].unsqueeze(-1)).sum(dim=0, keepdim=True)
        return (y + sh).view(1, 1, H)

    def forward(self, h, pos, ngram_emb=None):
        if self.use_ple and not _NO_PLE:
            h = h + self._ple(h, ngram_emb)
        if self.a_di is not None:
            mixed, inj = gated_residual(h, self.a_hc_norm, self.a_di, self.a_up, None, n_inj=HC)
        else:
            mixed, inj = gated_residual(h, self.a_hc_norm, self.a_down, self.a_up, self.a_inj)
        y = self._deltanet(mixed) if self.ltype in ("linear", "ple") else self._qsa(mixed, pos)
        h = inject(h, inj, y)
        if self.m_di is not None:
            mixed, inj = gated_residual(h, self.m_hc_norm, self.m_di, self.m_up, None, n_inj=HC)
        else:
            mixed, inj = gated_residual(h, self.m_hc_norm, self.m_down, self.m_up, self.m_inj)
        y = self._moe(mixed)
        return inject(h, inj, y)


def merge_scales_g64(q, s, gs_in=32, gs_out=64, to_int8=True, chunk=64):
    """专家权重的 scale 存储压缩：g32 fp16 → g64 + int8（每行 fp16 基准）。

    显存账：专家 scale 从 0.1465 GiB/层 降到 0.0403 GiB/层（−0.106 GiB/层 = −2.55 GiB/卡），
    这是 48 层能装进 2×60.5 GiB 的关键。代价：专家量化误差 10.02% →（实测）约 12%。
    返回 (q_int8, s_int8[E,N,K/gs_out], base_fp16[E,N])。
    """
    E, N, K = q.shape
    G = K // gs_out
    r = gs_out // gs_in
    out_q = torch.empty_like(q)                 # 非原地：原地写回在满载时会触发 MTE 越界
    s64 = torch.empty(E, N, G, dtype=torch.float16, device=q.device)
    for e0 in range(0, E, chunk):
        e1 = min(e0 + chunk, E)
        m = e1 - e0
        qc = q[e0:e1].float().view(m, N, G, r, gs_in)
        sc = s[e0:e1].float().view(m, N, G, r)
        # ★★ 第 36 轮：percentile 裁剪开关 `FNX_Q4_CLIP`
        #   ns 是新组（gs_out 个 k）的量化步长，可表示范围是 ±8·ns。
        #   取 ns = max(sc)（默认）时，正好覆盖组内最大幅值（不裁剪、步长最粗）；
        #   取 ns = CLIP·max(sc)（CLIP<1）时，步长变细、代价是削掉 |w| > 8·CLIP·max(sc) 的尾部。
        #   4-bit 只有 16 个电平，文献里 0.9–0.95 的裁剪通常能降 MSE。必须真重打包才能验。
        ns = (sc.amax(-1, keepdim=True) * _Q4_CLIP).clamp_min(1e-8)
        qn = torch.round(qc * (sc / ns).unsqueeze(-1)).clamp_(-8, 7).reshape(m, N, K)
        out_q[e0:e1] = qn.to(torch.int8)
        s64[e0:e1] = ns.squeeze(-1).to(torch.float16)
        del qc, sc, ns, qn
    if not to_int8:
        return out_q, s64, None
    base = (s64.float().amax(-1, keepdim=True) / 127.0).clamp_min(1e-12)
    s8 = torch.round(s64.float() / base).clamp_(-127, 127).to(torch.int8)   # base 已含 /127
    return out_q, s8, base.squeeze(-1).to(torch.float16)


_MOE_OPS = {}
# ★ 第 28 轮消融开关：PLE/ngram 是我们自己用 int64 重写的哈希查询，参考实现因 51 GB 表
# 无法实例化 → 这条路径从未与任何参考比对过，是"输出乱码"唯一剩下的嫌疑。
# FNX_NO_PLE=1  跳过整个 PLE 注入
# FNX_NO_NGRAM=1 不建 ngram 表（PLE 仍跑，但输入恒为 0）
_NO_PLE = os.environ.get("FNX_NO_PLE", "0") == "1"
# MoE 段落打点开关，默认关。★ 打印里带 idx[0].tolist() 会强制 device→host 同步，
# 实测 46 层 × 每层 3 次 = 138 次同步/token，把 TPOT 从 ~370 ms 抬到 465 ms，且 22 s 的
# prefill 无法入图。调测时才置 FNX_MOE_DBG=1。
_MOE_DBG = os.environ.get("FNX_MOE_DBG", "0") == "1"

def _get_moe_op(dev=None):
    """按设备缓存：W4GemvOp 内部缓存 _xe/_xo/_ids/_dbg 都是设备相关张量，
    跨卡复用会给出混合卡指针 -> aicore 507015（本轮实测定位）。"""
    key = str(dev)
    op = _MOE_OPS.get(key)
    if op is None:
        from .w4gemv_op import W4GemvOp
        op = W4GemvOp()
        _MOE_OPS[key] = op
    return op
