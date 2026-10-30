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
    """w 可以是 bf16 Tensor、QW（按需反量化），或 Raw（已反量化的包装器）。"""
    if isinstance(w, QW):
        return w(x)
    if isinstance(w, Raw):
        return _fl(x, w.w)
    return _fl(x, w)

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
        self.k = [torch.zeros(B, KVH, P, HD, dtype=torch.bfloat16, device=device) for _ in range(n)]
        self.v = [torch.zeros(B, KVH, P, HD, dtype=torch.bfloat16, device=device) for _ in range(n)]


def gated_residual(h, hc_norm_w, w_down, w_up, w_inj):
    b = h.shape[0]
    hn = rms_norm(h, hc_norm_w, gs=H)
    mix = F.silu(lin(hn, w_down) / HC)
    mix = torch.sigmoid(lin(mix, w_up))
    mix = mix.view(b, 1, HC, H)
    mixed = (mix * hn.view(b, 1, HC, H)).mean(dim=-2).view(b, 1, H)
    if w_inj is None:
        return mixed
    inj = 2.0 * torch.sigmoid(lin(hn, w_inj) / HC)
    return mixed, inj.view(b, 1, HC)


def inject(hyper, inj, y):
    b = hyper.shape[0]
    return hyper + (y.view(b, 1, 1, H) * inj.unsqueeze(-1)).reshape(b, 1, HCH)


class FnLayer:
    def __init__(self, idx, ltype, W, device):
        self.idx = idx
        self.ltype = ltype
        self.dev = device
        self.kv = None
        self.ki = 0
        self.inv = inv_freq(device)
        self.use_ple = ltype == "ple"

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

    # ---------- DeltaNet 单步 ----------
    def _deltanet(self, h):
        b = h.shape[0]
        mq = lin(h, self.w_qkv).view(b, CONV_DIM)
        buf = torch.cat([self.conv, mq.unsqueeze(-1)], dim=-1)              # [b,D,4]
        y = (buf * self.conv_w.unsqueeze(0)).sum(-1)
        y = F.silu(y)
        self.conv.copy_(buf[..., 1:])
        q, k, v = torch.split(y, [KEY_DIM, KEY_DIM, VAL_DIM], dim=-1)
        z = lin(h, self.w_z).view(b, NVH, HVD)
        beta = torch.sigmoid(lin(h, self.w_b)).view(b, NVH, 1)
        a = lin(h, self.w_a)
        g = (-torch.exp(self.A_log) * F.softplus(a.float() + self.dt_bias)).view(b, NVH)

        rep = NVH // NKH
        q = l2norm(q.view(b, NKH, HKD)).repeat_interleave(rep, dim=1) / math.sqrt(HKD)
        k = l2norm(k.view(b, NKH, HKD)).repeat_interleave(rep, dim=1)
        v = v.view(b, NVH, HVD)

        S = self.S * torch.exp(g)[..., None, None]                          # 先衰减
        kv_mem = (S * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta
        S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
        o = (S * q.unsqueeze(-1)).sum(dim=-2)
        self.S.copy_(S)

        o = o.to(torch.bfloat16)
        xf = o.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
        xf = self.dn_norm_w * xf.to(torch.bfloat16)
        xf = xf * torch.sigmoid(z.float())          # output_gate_type = "sigmoid"（不是 silu）
        return lin(xf.to(torch.bfloat16).view(b, VAL_DIM), self.w_out)

    # ---------- QSA（稠密 causal，≤2048 上下文与参考一致） ----------
    def _qsa(self, h, pos):
        b = h.shape[0]
        cos, sin = rope_cs(pos, self.inv)
        if b > 1:
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)     # [b,1,64] 便于广播
        qg = lin(h, self.w_q).view(b, NAH, 2 * HD)
        q = rms_norm(qg[..., :HD], self.qn_w)
        gate = qg[..., HD:]
        k = rms_norm(lin(h, self.w_k).view(b, KVH, HD), self.kn_w)
        v = lin(h, self.w_v).view(b, KVH, HD)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        P = self.kv.P
        # 逐行写各自的 pos（one-hot 形式，图捕获已验证可用；比 index_put 稳）
        onehot = F.one_hot(pos, P).to(torch.bfloat16).view(b, 1, P, 1)      # [b,1,P,1]
        kc, vc = self.kv.k[self.ki], self.kv.v[self.ki]
        kc.copy_(kc * (1.0 - onehot) + k.view(b, KVH, 1, HD) * onehot)
        vc.copy_(vc * (1.0 - onehot) + v.view(b, KVH, 1, HD) * onehot)

        mask = (torch.arange(P, device=self.dev)[None, :] <= pos.view(b, 1)) \
            .view(b, 1, 1, P)
        Kk = kc.repeat_interleave(NAH // KVH, dim=1)                        # [b,NAH,P,HD]
        Vv = vc.repeat_interleave(NAH // KVH, dim=1)
        att = torch.matmul(q.view(b, NAH, 1, HD), Kk.transpose(2, 3)) * (HD ** -0.5)
        att = att.masked_fill(~mask, NEG)
        att = torch.softmax(att.float(), dim=-1).to(torch.bfloat16)
        o = torch.matmul(att, Vv).view(b, NAH * HD)
        return lin(o * torch.sigmoid(gate.reshape(b, NAH * HD)), self.w_o)

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
        sh = lin(F.silu(lin(x2, self.sh_gate)) * lin(x2, self.sh_up), self.sh_down)
        sh = torch.sigmoid(lin(x2, self.sh_gate2)) * sh
        logits = lin(x2, self.gate_w)
        probs = torch.softmax(logits.float(), dim=-1)
        w, idx = torch.topk(probs, TOPK, dim=-1)                 # [b,10]
        w = (w / w.sum(dim=-1, keepdim=True)).to(torch.bfloat16)

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
        mixed, inj = gated_residual(h, self.a_hc_norm, self.a_down, self.a_up, self.a_inj)
        y = self._deltanet(mixed) if self.ltype in ("linear", "ple") else self._qsa(mixed, pos)
        h = inject(h, inj, y)
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
        ns = sc.amax(-1, keepdim=True).clamp_min(1e-8)
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
