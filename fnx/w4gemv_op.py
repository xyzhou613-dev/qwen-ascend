"""W4A16 GEMV 算子的接入层：把 AscendC kernel（libfnxw4gemv.so）包成 model.py 能用的接口。

设计要点（每条都有实测依据，见 deploy/...第27轮 §7–§8）：

1. **s4 必须在装配期转置**成 [K/gs][N]。实测：装配期一次性 0.02 s；每 token 现转 4.70 ms
   = kernel 专家段 21.8 ms 的 21.6%。所以本类把转置放在 `prepare_layer()`（装配期调一次），
   `run()` 里绝不做转置。

2. **路由收拢**：kernel 按 `expertIdx * stride` 线性寻址，要求 10 个激活专家的 p4/s4T 在
   设备上连续。真链路 top-10 是散落的，两种解法：
     (a) 每次 decode 现让 NPU 做 device→device gather（10×1.6 MB/层）
     (b) 给 kernel 加一张 int32[10] 的专家索引表（更省，但要改 kernel）
   本模块实现 (a) 作为可用路径，(b) 记在 TODO 里。

3. **输出**是 [nExp][N]，加权求和（Σ w_e·y_e）在 Python 侧做，量级 10×N float，可忽略。

用法：
    op = W4GemvOp(so_path="/root/ascend-dev/ascendc/direct/libfnxw4gemv.so")
    s4t = op.prepare_layer(s4_fp16_np)             # 装配期一次：[E][N][K/gs] -> 设备上的 [E][K/gs][N]
    y   = op.run(x_np, p4_np, s4t_local, ids)      # decode 每 token
"""
import ctypes
import os

import numpy as np
import torch
import torch_npu  # noqa: F401

GS = 32
SO_DEFAULT = "/root/ascend-dev/ascendc/direct/libfnxw4gemv.so"
# 调测开关：1 = 每次裸 acl 发射后立刻同步本设备（数值/故障定位用），0 = 生产（默认）
_SYNC = os.environ.get("FNX_KERNEL_SYNC", "0") == "1"
# 审计开关：首次发射时打印各指针的真实设备号（跨卡问题必须靠打点，不能靠猜）
_AUDIT = os.environ.get("FNX_KERNEL_AUDIT", "0") == "1"


class W4GemvOp:
    def __init__(self, so_path=SO_DEFAULT, nblk=20, dbg_elems=65536, device=None):
        if not os.path.exists(so_path):
            raise FileNotFoundError(so_path)
        self.lib = ctypes.CDLL(so_path)
        self.lib.fnx_w4gemv_launch.restype = ctypes.c_int
        self.lib.fnx_w4gemv_launch.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 5 + [ctypes.c_void_p]
        self.nblk = nblk
        self._dev = device if device is not None else "npu:0"
        self._dbg_elems = dbg_elems
        # _dbg 必须**按设备**分配：早先固定成 npu:0，跨卡时 dev1 的 kernel 会拿到 dev0 的指针
        self._dbg_by_dev = {}
        self._audited = set()
        self._xe = self._xo = None
        self._shape = None
        self._ids = None

    def _dbg_for(self, dev):
        t = self._dbg_by_dev.get(str(dev))
        if t is None:
            t = torch.zeros(self._dbg_elems, dtype=torch.float32, device=dev)
            self._dbg_by_dev[str(dev)] = t
        return t

    # ---------- 装配期 ----------
    def prepare_layer(self, s4_np):
        """s4_np: [E][N][K/gs] fp16 (numpy) -> 设备上 [E][K/gs][N] fp16 tensor。

        实测代价：单层 0.49 ms（gate_up+down 合计），只做一次，不进 decode 路径。
        """
        s4t = np.ascontiguousarray(np.transpose(s4_np, (0, 2, 1)), dtype=np.float16)
        return torch.from_numpy(s4t).npu()

    @staticmethod
    def _split_x(x_np):
        """x: [E][K] float32 -> 偶/奇 lane 两份连续 [E][K/2]"""
        xe = np.ascontiguousarray(x_np[:, 0::2], dtype=np.float32)
        xo = np.ascontiguousarray(x_np[:, 1::2], dtype=np.float32)
        return xe, xo

    # ---------- decode 每 token ----------
    def run(self, p4_dev, s4t_dev, x_dev, n_exp, N, K, expert_ids=None, lay=1, out=None):
        """p4_dev: npu int8 [E][N][K/2]（连续，已按需 gather 好）
           s4t_dev: npu fp16 [E][K/gs][N]（装配期做好）
           x_dev:  npu float32 [E][K]
        返回 npu float32 [E][N]
        """

        xe_np, xo_np = None, None
        # x 的偶/奇拆分：K 很小（≤2560），在这里做是 10KB 级，可忽略
        # 注意：不做按形状缓存！之前按 (nE,N,K) 缓存 xe/xo，同一形状下换 x 会静默复用旧数据
        # （selftest 里 moe_forward maxerr 0.25 就是这个 bug）。拆分是 [nE][K] 级，量级很小。
        if isinstance(x_dev, np.ndarray):
            if lay == 2:
                xn = np.ascontiguousarray(x_dev, dtype=np.float32)   # v2：自然 x，不拆
                self._xe = torch.from_numpy(xn).npu()
                self._xo = self._xe
            else:
                xe_np, xo_np = self._split_x(x_dev)
                self._xe = torch.from_numpy(xe_np).npu()
                self._xo = torch.from_numpy(xo_np).npu()
        else:
            # 设备侧拆分：零宿主同步（x 只有 nE*K 个元素，strided contiguous 成本可忽略；
            # 千万不要为了拆分做 .cpu()，那会每层每张量一次同步往返）
            xt = x_dev if (x_dev.dtype == torch.float32 and x_dev.is_contiguous()) else x_dev.float().contiguous()
            if lay == 2:
                self._xe = xt.contiguous()          # v2：直接吃自然 x，kernel 内取两段连续块
                self._xo = xt
            else:
                self._xe = xt[:, 0::2].contiguous()
                self._xo = xt[:, 1::2].contiguous()
        self._shape = (n_exp, N, K)

        # 设备号必须在最前面定出来：后面 _ids 的兜底分配也要用它（曾经在下面才定义 →
        # 走 expert_ids=None 分支时 NameError）。
        _dev = x_dev.device if isinstance(x_dev, torch.Tensor) else self._dev
        if isinstance(expert_ids, torch.Tensor) and expert_ids.device.type == "npu":
            _dev = expert_ids.device

        if expert_ids is None:
            if self._ids is None or self._ids.numel() != n_exp or self._ids.device != _dev:
                self._ids = torch.arange(n_exp, dtype=torch.int32, device=_dev)
            ids_dev = self._ids
        elif isinstance(expert_ids, torch.Tensor):
            # 支持设备张量（model.py 里 idx[0] 是设备张量，直接转 numpy 会报错）
            ids_dev = expert_ids.to(torch.int32)
            if ids_dev.device != _dev:
                ids_dev = ids_dev.to(_dev)
        else:
            ids_dev = torch.from_numpy(np.ascontiguousarray(expert_ids, dtype=np.int32)).to(_dev)
        _ydev = _dev
        y = out if out is not None else torch.empty((n_exp, N), dtype=torch.float32, device=_ydev)
        # 跨设备指针检查：kernel 拿到别的卡上的地址 = 要么 507015，要么静默读垃圾。
        # 宁可在这里报一句清楚的话，也不要等到 model.py:220 的同步点。
        for _nm, _t in (("p4", p4_dev), ("s4t", s4t_dev), ("ids", ids_dev)):
            if isinstance(_t, torch.Tensor) and _t.device != _ydev:
                raise RuntimeError(
                    f"w4gemv 设备不一致: {_nm} 在 {_t.device}，而 x/y 在 {_ydev}"
                    f"（kernel 不支持跨设备指针，请在装配期把权重放到层所在设备）")
        _dbg = self._dbg_for(_ydev)
        if _AUDIT and str(_ydev) not in self._audited:
            self._audited.add(str(_ydev))
            print(f"[w4gemv-audit] dev={_ydev} x={getattr(x_dev, 'device', None)} "
                  f"p4={getattr(p4_dev, 'device', None)} s4t={getattr(s4t_dev, 'device', None)} "
                  f"ids={ids_dev.device} y={y.device} dbg={_dbg.device} N={N} K={K} nE={n_exp}",
                  flush=True)
        # ★ 必须用 **本设备** 的 stream：裸 acl 发射带的是 stream handle，跨设备拿 dev0 的
        # stream 去发 dev1 的指针，是 §35 里 507015 的第一嫌疑（AICORE 检查位）。
        st = 0
        try:
            st = int(torch.npu.current_stream(_ydev).npu_stream)
        except Exception:  # noqa: BLE001
            try:
                st = int(torch.npu.current_stream().npu_stream)
            except Exception:  # noqa: BLE001
                st = 0
        rc = self.lib.fnx_w4gemv_launch(
            ctypes.c_void_p(p4_dev.data_ptr()), ctypes.c_void_p(s4t_dev.data_ptr()),
            ctypes.c_void_p(self._xe.data_ptr()), ctypes.c_void_p(self._xo.data_ptr()),
            ctypes.c_void_p(y.data_ptr()), ctypes.c_void_p(_dbg.data_ptr()),
            ctypes.c_void_p(ids_dev.data_ptr()),
            int(N), int(K), int(n_exp), int(self.nblk), int(lay), ctypes.c_void_p(st))
        if rc != 0:
            raise RuntimeError(f"fnx_w4gemv_launch rc={rc}")
        # 裸 acl 发射**不受** ASCEND_LAUNCH_BLOCKING 保护，出错会在下一个 torch 同步点
        # （model.py:220 self.pos0.add_）才炸，现场全毁。调测时置 FNX_KERNEL_SYNC=1，
        # 让 aicore 错立刻落在这次 launch 上。
        if _SYNC:
            torch.npu.synchronize(_ydev)
        return y

    # ---------- 批量（一次过桥发射 24 层） ----------
    def run_layers(self, p4_list, s4t_list, x_list, y_list, N, K, n_exp, lay=2):
        """一次 ctypes 调用发射 len(p4_list) 个 launch。

        dict/列表的每个元素都是**设备张量**；y_list 是预分配的输出张量。
        收益：把 96 次 Python→C 过桥（~35 us/次）压成 1 次。
        """
        nL = len(p4_list)
        if nL == 0:
            return y_list
        def arr(ts):
            return (ctypes.c_void_p * nL)(*[ctypes.c_void_p(t.data_ptr()) for t in ts])
        _dev = p4_list[0].device
        st = 0
        try:
            st = int(torch.npu.current_stream(_dev).npu_stream)
        except Exception:  # noqa: BLE001
            try:
                st = int(torch.npu.current_stream().npu_stream)
            except Exception:  # noqa: BLE001
                st = 0
        rc = self.lib.fnx_w4gemv_launch_layers(arr(p4_list), arr(s4t_list), arr(x_list), arr(y_list),
                                               nL, int(N), int(K), int(n_exp), int(self.nblk), int(lay),
                                               ctypes.c_void_p(self._dbg_for(_dev).data_ptr()),
                                               ctypes.c_void_p(st))
        if rc != 0:
            raise RuntimeError(f"fnx_w4gemv_launch_layers rc={rc}")
        if _SYNC:
            torch.npu.synchronize(_dev)
        return y_list

    def make_x_dev(self, x_np, lay=2):
        """把 numpy x 变成 kernel 需要的设备张量（v2 直接就是自然 x）。"""
        if lay == 2:
            return torch.from_numpy(np.ascontiguousarray(x_np, dtype=np.float32)).npu()
        xe, xo = self._split_x(np.asarray(x_np, dtype=np.float32))
        return torch.from_numpy(xe).npu()
    # ---------- MoE 整段（model.py 直接换这个） ----------
    def moe_forward(self, p4_dev, s4t_dev, x_np, topk_ids, topk_w, N, K):
        """单 token 的 MoE 专家段，替代原来的 `ExpQ` 路径（torch 反量化 + bmm）。

        p4_dev  : 设备上 U8，按**真实专家号**线性寻址（[E_tot][N][K/2]），不需要 gather
        s4t_dev : 设备上 fp16 [nE][K/gs][N]，**装配期**由 prepare_layer() 做好，按槽位
        x_np    : numpy float32 [K]，这一 token 的 MoE 输入
        topk_ids: 长度 nE 的专家号（router 给的 top-k）
        topk_w  : 长度 nE 的权重（router 概率，已是 float32）
        返回      : 设备上 float32 [N]（= Σ_e w_e·y_e）
        """
        nE = len(topk_ids)
        x_b = np.repeat(np.asarray(x_np, dtype=np.float32).reshape(1, K), nE, axis=0)
        y = self.run(p4_dev, s4t_dev, x_b, nE, N, K,
                     expert_ids=np.asarray(topk_ids, dtype=np.int32))
        return self.weighted_sum(y, topk_w)
    @staticmethod
    def to_v2_layout(p4_old, N, K):
        """把旧布局 p4（byte j: lo=W[2j], hi=W[2j+1]）转成 v2 布局。

        v2: byte(64s+j) 低4bit = W[128s+j]，高4bit = W[128s+64+j]
        已由 tools/layout_v2_check.py 验证：无损 + 数值恒等（max_abs 8.9e-16）。
        p4_old 形状 [nE][N][K/2] 或 [N][K/2]。
        """
        squeeze = (p4_old.ndim == 2)
        a = p4_old[None] if squeeze else p4_old       # [nE][N][X]
        nE, Nx, X = a.shape
        if X == K:
            # int8 容器格式（每字节一个 -8..7 的值；服务 FNX_PACK=0 用的就是这个）
            w = a.reshape(nE * Nx, K).astype(np.int8)
        else:
            assert X == K // 2, (a.shape, N, K)
            b = a.reshape(nE * Nx, K // 2).astype(np.int8).view(np.uint8)
            lo = ((b & 0x0F).astype(np.int8) ^ 8) - 8
            hi = (((b >> 4) & 0x0F).astype(np.int8) ^ 8) - 8
            w = np.empty((nE * Nx, K), dtype=np.int8)
            w[:, 0::2] = lo
            w[:, 1::2] = hi
        out = np.zeros((nE * N, K // 2), dtype=np.uint8)
        for s in range(K // 128):
            il = slice(128 * s, 128 * s + 64)
            ih = slice(128 * s + 64, 128 * s + 128)
            out[:, 64 * s:64 * s + 64] = ((w[:, il].astype(np.int16) & 0x0F) |
                                          ((w[:, ih].astype(np.int16) & 0x0F) << 4)).astype(np.uint8)
        res = out.reshape(nE, N, K // 2)
        return res[0] if squeeze else res
    @staticmethod
    def weighted_sum(y_dev, w):
        """Σ_e w_e·y_e。

        ★ w 强烈建议传**设备张量**：传 numpy 会每次做一次 H2D，实测让 24 层从
        14.92 ms 涨到 25.38 ms（+41%，见第 27 轮 §21.2）。decode 循环里必须用设备张量。
        """
        if isinstance(w, torch.Tensor):
            wt = w.to(torch.float32) if w.dtype == torch.float32 else w.to(torch.float32)
            if wt.device.type != "npu":
                wt = wt.npu()
        else:
            wt = torch.from_numpy(np.asarray(w, dtype=np.float32)).npu()
        return (y_dev * wt.view(-1, 1)).sum(dim=0)


def _bench_split():
    """量设备侧 x 拆分成本（真链路每层每张量都要做一次）。"""
    import time
    for name, nE, K in (("gate_up", 10, 2560), ("down", 10, 640)):
        x = torch.randn(nE, K, dtype=torch.float32, device="npu:0")
        for _ in range(3):
            x[:, 0::2].contiguous(); x[:, 1::2].contiguous()
        torch.npu.synchronize()
        t0 = time.perf_counter()
        for _ in range(20):
            xe = x[:, 0::2].contiguous(); xo = x[:, 1::2].contiguous()
        torch.npu.synchronize()
        ms = (time.perf_counter() - t0) / 20 * 1e3
        print(f"  split {name:8s} [10,{K}] = {ms*1000:.1f} us -> x96/token = {ms*96:.2f} ms", flush=True)
def _selftest():
    """用 golden multi 数据跑一遍接入层，确认与 numpy 契约一致。"""
    import json
    GOLD = "/root/ascend-dev/golden/w4gemv"
    meta = json.load(open(os.path.join(GOLD, "meta.json")))
    op = W4GemvOp()
    for name in ("multi_gate_up_proj", "multi_down_proj"):
        c = meta["cases"][name]
        base = name.replace("multi_", "")
        d = os.path.join(GOLD, c["dir"])
        nE, N, K = c["B"], c["N"], c["K"]
        p4 = np.fromfile(os.path.join(d, base + "_p4.bin"), dtype=np.int8).reshape(nE, N, K // 2)
        s4 = np.fromfile(os.path.join(d, base + "_s4.bin"), dtype=np.float16).reshape(nE, N, K // GS)
        x = ((np.fromfile(os.path.join(d, base + "_x.bin"), dtype=np.int16)
              .astype(np.uint32) << 16).view(np.float32)).reshape(nE, K)

        s4t_dev = op.prepare_layer(s4)                       # ← 装配期（一次）
        p4_dev = torch.from_numpy(p4).npu()
        y = op.run(p4_dev, s4t_dev, x, nE, N, K)             # ← decode
        got = y.cpu().numpy()

        b = p4.view(np.uint8)
        lo = ((b & 0x0F).astype(np.int8) ^ 8) - 8
        hi = (((b >> 4) & 0x0F).astype(np.int8) ^ 8) - 8
        w4 = np.empty((nE, N, K), dtype=np.int8)
        w4[:, :, 0::2] = lo
        w4[:, :, 1::2] = hi
        ref = np.einsum("enk,ek->en", w4.astype(np.float32) *
                        np.repeat(s4.astype(np.float32), GS, axis=2), x)
        d_ = np.abs(got - ref)
        rel = float(d_.mean() / (np.abs(ref).mean() + 1e-12))
        tol = meta["tolerance"]["rel_mean"]
        print(f"  {name:22s} rel_mean {rel*100:.4f}%  -> {'PASS' if rel <= tol else 'FAIL'}", flush=True)
        # 索引表验证：给逆序 expert_ids，则 y[e] 必须 = Σ_k W_{perm[e]}[n,k]·s4[e]·x[e]
        # 注意：kernel 里 p4 按 expert_ids 取，但 s4/x 按槽位取 → 参考必须是 w4[perm]*s4[槽位]
        wf = w4.astype(np.float32) * np.repeat(s4.astype(np.float32), GS, axis=2)
        perm = np.arange(nE - 1, -1, -1, dtype=np.int32)
        got2 = op.run(p4_dev, s4t_dev, x, nE, N, K, expert_ids=perm).cpu().numpy()
        e2 = float(np.abs(got2 - np.einsum("enk,ek->en", (w4.astype(np.float32) * np.repeat(s4.astype(np.float32), GS, axis=2))[perm], x)).max())
        print(f"  {'':22s} 索引表(逆序) maxerr {e2:.3e} -> {'PASS' if e2 < 1e-5 else 'FAIL'}", flush=True)
        # moe_forward 验证：单 token 广播给 10 个专家 + 加权求和
        ids = np.random.RandomState(1).permutation(nE)[:nE].astype(np.int32)
        w = np.random.RandomState(2).rand(nE).astype(np.float32)
        out = op.moe_forward(p4_dev, s4t_dev, x[0], ids, w, N, K).cpu().numpy()
        wf2 = w4.astype(np.float32) * np.repeat(s4.astype(np.float32), GS, axis=2)
        ref_moe = (np.einsum("enk,k->en", (w4.astype(np.float32) * np.repeat(s4.astype(np.float32), GS, axis=2))[ids], x[0]) * w[:, None]).sum(0)
        e3 = float(np.abs(out - ref_moe).max())
        print(f"  {'':22s} moe_forward maxerr {e3:.3e} -> {'PASS' if e3 < 1e-5 else 'FAIL'}", flush=True)
        # v2 布局验证：p4 转 v2 + 自然 x（不拆分）→ 必须与同一份参考一致
        p4v2 = op.to_v2_layout(p4, N, K)
        p4v2_dev = torch.from_numpy(np.ascontiguousarray(p4v2, dtype=np.uint8)).npu()
        yv2 = op.run(p4v2_dev, s4t_dev, x, nE, N, K, lay=2).cpu().numpy()
        ev2 = float(np.abs(yv2 - ref).max())
        print(f"  {'':22s} v2 布局 maxerr {ev2:.3e} -> {'PASS' if ev2 < 1e-5 else 'FAIL'}", flush=True)
        # 加权求和 smoke test
        w = np.random.RandomState(0).rand(nE).astype(np.float32)
        s_dev = op.weighted_sum(y, w)
        s_ref = (ref * w[:, None]).sum(0)
        print(f"  {'':22s} weighted_sum maxerr {np.abs(s_dev.cpu().numpy()-s_ref).max():.3e}", flush=True)


if __name__ == "__main__":
    _bench_split()
    _selftest()
