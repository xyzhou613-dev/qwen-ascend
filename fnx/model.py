"""48 层装配 + graph-safe 步进 + 图捕获。"""
import json
import os
import time

import torch
import torch.nn.functional as F

from . import engine as E
from .load import QDIR, SRCDIR, NgramTable, load_layer, load_nonlayer

MASK64 = (1 << 64) - 1
SIGN64 = 1 << 63
# dev1 capture is opt-in. Always pass its stream explicitly to graph():
# the process-wide default capture stream may belong to dev0.
_CAP_G1 = os.environ.get("FNX_GRAPH_G1", "0")
# 从第几层开始把专家权重 offload 到宿主锁页（默认 999 = 不 offload）
_OFFLOAD_FROM = int(os.environ.get("FNX_OFFLOAD_FROM", "999"))
# Capture the mixer and lm_head on an explicit stream of head_dev.
_CAP_GH = os.environ.get("FNX_GRAPH_HEAD", "0") == "1"
# "0"=不捕获 dev1；"1"=整段 body1 捕获（实测跨设备重放无效）；"layer"=**每层一张单设备图**
# 每层一张图是绕开"NPUGraph 跨设备段无效"的关键：图只含 dev1 上的单层内核，
# 层与层之间用 `self.h1` 这个固定缓冲区串起来，地址稳定，适合入图。


def _s64(v):
    v &= MASK64
    return v - (1 << 64) if v >= SIGN64 else v


class FnModel:
    def __init__(self, P=2048, lpc=24, dev0="npu:0", dev1="npu:1", verbose=True,
                 nlayers=None, with_head=True, with_ngram=True, layer_start=0):
        cfg = json.load(open(os.path.join(SRCDIR, "config.json")))["text_config"]
        if os.environ.get("FNX_NO_NGRAM") == "1":     # ★消融开关，见 engine._NO_PLE 注释
            with_ngram = False
        self.lt = cfg["layer_types"]
        self.n = len(self.lt) if nlayers is None else nlayers
        self.P = P
        self.lpc = lpc
        self.dev0, self.dev1 = dev0, dev1
        self.verbose = verbose
        self.g0 = self.g1 = self.gh = None
        self.layers0, self.layers1 = [], []
        self.with_head = with_head
        dev_h1 = dev1 if self.n > lpc else dev0

        lt_ = self.lt[: self.n]
        n_full0 = sum(1 for t in lt_[layer_start:lpc] if t in E.FULL_TYPES)
        n_full1 = sum(1 for t in lt_[layer_start:] if t in E.FULL_TYPES) - n_full0
        self.kv0 = E.KVStore(P, dev0, n_full0)
        self.kv1 = E.KVStore(P, dev_h1, n_full1)

        t0 = time.time()
        # ★ head 的安置：默认放 **dev1**（有 2 层以上 eager 时）。
        #   理由（第 30 轮实测）：dev0 已吃到 58.85/60.96 GiB，LPC=46 只差 82 MiB 就 OOM；
        #   把 int8 lm_head（约 646 MiB）+ mixer 搬到 dev1 就腾出这一层的位置，
        #   TPOT 154.4 → ~142 ms。附带好处：layers1 跑完后隐藏状态本来就在 dev1，
        #   head 在 dev1 算可以**省掉每 token 两次跨卡中转**。
        _hd = os.environ.get("FNX_HEAD_DEV", "")
        if _hd in ("0", "1"):
            head_dev = dev0 if _hd == "0" else dev1
        else:
            head_dev = dev1 if self.n > lpc else dev0
        self.head_dev = head_dev
        (self.emb_q, self.emb_s), lm_head, mx = load_nonlayer(dev0, head_dev,
                                                              with_lmhead=with_head)
        # ★ 装载期预反量化 lm_head（606 MiB int8 → 1.18 GiB bf16）：每 token 只算一次 GEMM，
        #   不再分 16 块重复反量化（实测该路径 7.51 ms/token）。值逐位相同。
        #   dev1 空闲显存充足，这一步不占 dev0。
        if (E._DEQ_HEAD and hasattr(lm_head, "deq_w")
                and not getattr(lm_head, "chunks", None)):
            _t = time.time()
            self.lm_head = E.Raw(lm_head.deq_w())
            del lm_head
            self._log(f"lm_head 预反量化 {time.time()-_t:.1f}s")
        else:
            self.lm_head = lm_head
        self.mix_norm = mx["hc_norm.weight"]
        self.mix_down = mx["input_mix_weight_down.weight"]
        self.mix_up = mx["input_mix_weight_up.weight"]
        self._log(f"nonlayer 完成 {time.time()-t0:.1f}s")

        i0 = i1 = 0
        n_off = 0
        off_bytes = 0
        for i in range(layer_start, self.n):
            on0 = i < lpc
            dev = dev0 if on0 else dev1
            ltype = "ple" if i == E.PLE_LAYER else ("full" if lt_[i] in E.FULL_TYPES else "linear")
            ta = time.time()
            W = load_layer(os.path.join(QDIR, f"L{i:02d}.safetensors"), i, dev)
            # ★ offload：把第 >= FNX_OFFLOAD_FROM 层的专家权重搬到宿主锁页，
            # 每 token 按路由结果逐专家连续 H2D（实测 21.4 GB/s）。省 2.34 GiB/层。
            if i >= _OFFLOAD_FROM:
                for _k in ("mlp.experts.gate_up_proj", "mlp.experts.down_proj"):
                    _e = W.get(_k)
                    if _e is not None and hasattr(_e, "to_host_offload") and _e.to_host_offload():
                        off_bytes += _e.q_off.numel()
                n_off += 1
            tb = time.time()
            L = E.FnLayer(i, ltype, W, dev)
            del W
            tc = time.time()
            if ltype == "full":
                if L.index_enabled:
                    L.index_k = torch.zeros(P, 128, dtype=torch.bfloat16, device=dev)
                if on0:
                    L.kv, L.ki = self.kv0, i0
                    i0 += 1
                else:
                    L.kv, L.ki = self.kv1, i1
                    i1 += 1
            (self.layers0 if on0 else self.layers1).append(L)
            if self.verbose:
                self._log(f"  layer {i:2d} type={ltype:5s} load {tb-ta:6.1f}s init {tc-tb:5.1f}s "
                          f"累计 {time.time()-t0:6.1f}s  "
                          f"hbm0={torch.npu.memory_allocated(self.dev0)/2**30:5.2f}GiB "
                          f"hbm1={torch.npu.memory_allocated(self.dev1)/2**30:5.2f}GiB")
            if True:
                torch.npu.empty_cache()

        # ★ 第31轮：所有"逐序列"缓冲区都带批维 B（见 engine.B）；B=1 时形状与改造前一致。
        _B = E.B
        self.tok = torch.zeros(_B, dtype=torch.int64, device=dev0)
        self.pos0 = torch.zeros(_B, dtype=torch.int64, device=dev0)
        self.pos1 = torch.zeros(_B, dtype=torch.int64, device=dev_h1)
        self.ngram_emb = torch.zeros(_B, 1, E.H, dtype=torch.bfloat16, device=dev0)
        self.emb_buf = torch.zeros(_B, 1, E.H, dtype=torch.bfloat16, device=dev0)
        # 分块 prefill 用的 token 缓冲（第 46 轮）：[CHUNK,1,H]
        self.emb_chunk = torch.zeros(max(1, E._CHUNK), 1, E.H, dtype=torch.bfloat16, device=dev0)
        self.h0 = torch.zeros(_B, 1, E.HCH, dtype=torch.bfloat16, device=dev0)
        self.h1 = torch.zeros(_B, 1, E.HCH, dtype=torch.bfloat16, device=dev_h1)
        self.logits = torch.zeros(_B, E.VOCAB, dtype=torch.bfloat16, device=head_dev)
        self.ctx = [E.NG_EOS] * E.NG_CTX            # 仅 B=1 的 n-gram 路径用
        self._host_pos = 0
        self.g0_short = None
        self.g1_short = None
        self.h_host = torch.zeros(_B, 1, E.HCH, dtype=torch.bfloat16)  # cross-device staging (P2P unavailable)
        self.table = None
        # ★ 图捕获用的旁路 stream。replay 必须回到**同一条 stream** 上：
        # 否则 dev0→host→dev1 的中转拷贝（默认流）与 g1 replay（旁路流）之间没有顺序保证，
        # dev1 的图会读到上一个 token 的陈旧 h1（实测 16 层 8+8 时 A vs C 差 54.169%）。
        self._gs = None
        self._gs1 = None
        self._g1_list = None
        if with_ngram:
            self.table = NgramTable()
            from safetensors.torch import load_file
            sd = load_file(os.path.join(QDIR, f"L{E.PLE_LAYER:02d}.safetensors"))
            pe = f"model.language_model.layers.{E.PLE_LAYER}.ple.ple_embedding."
            self.mult = [int(x) for x in sd[pe + "layer_multipliers"].tolist()]
            self.head_vocab = [int(x) for x in sd[pe + "ngram_heads_vocab_sizes"].tolist()]
            self.head_off = [int(x) for x in sd[pe + "ngram_heads_offsets"].tolist()]
            del sd
        self._log(f"装配完成 {time.time()-t0:.1f}s  "
                  f"hbm0={torch.npu.memory_allocated(self.dev0)/2**30:.2f}GiB "
                  f"hbm1={torch.npu.memory_allocated(self.dev1)/2**30:.2f}GiB")
        if n_off:
            self._log(f"[offload] {n_off} 层的专家权重在宿主锁页：{off_bytes/2**30:.1f} GiB")

    def _log(self, s):
        if self.verbose:
            print(s, flush=True)

    # ---------------- 三段 graph-safe ----------------
    def body0(self):
        h = self.emb_buf.repeat(1, 1, E.HC)
        for L in self.layers0:
            h = L.forward(h, self.pos0, self.ngram_emb)
        self.h0.copy_(h)

    def body1(self):
        if not self.layers1:
            return
        # ★★ 必须显式切到 dev1 的设备上下文（第 33 轮实测）：
        #   eager 下当前设备仍是 0，而张量在 dev1 → **每个算子都要做一次设备切换**。
        #   实测单层 13.58 → 7.85 ms（−5.74 ms/层，3 层 ≈ −17.2 ms/token）。
        with torch.npu.device(self.dev1):
            h = self.h1
            for L in self.layers1:
                h = L.forward(h, self.pos1)
        self.h1.copy_(h)

    def host_lmhead(self, x):
        """宿主侧 lm_head：int8 常驻 DDR，按 N 分块，分组反量化后求和。x: CPU fp32 [K]。"""
        q, s, pad = self.lm_head
        V, K = q.shape
        gs = K // s.shape[-1]
        xg = x.view(K // gs, gs)
        out = torch.empty(V, dtype=torch.float32)
        for n0 in range(0, V, 32768):
            n1 = min(n0 + 32768, V)
            qc = q[n0:n1].view(n1 - n0, K // gs, gs).float()
            out[n0:n1] = (torch.einsum("ngk,gk->ng", qc, xg) * s[n0:n1].float()).sum(-1)
        return out

    def _head_device(self):
        """head 的**纯 device 部分**：mixer + lm_head（都在 self.head_dev 上）。

        ★ 同样要显式切设备上下文：head 在 dev1 时，未切上下文实测 8.74 → 5.66 ms。
        """
        h = self.h0 if self.head_dev == self.dev0 else self.h1
        b = h.shape[0]
        with torch.npu.device(self.head_dev):
            m = E.gated_residual(h, self.mix_norm, self.mix_down, self.mix_up, None)
            self.logits.copy_(E.lin(m.view(b, E.H), self.lm_head))

    def head(self):
        _dbg = os.environ.get("FNX_MOE_DBG", "0") == "1"
        _fp = os.environ.get("FNX_HEAD_DBG", "0") == "1"
        if _dbg: print(f"[head] enter h1.dev={self.h1.device} h_host.dev={self.h_host.device}", flush=True)
        if not self.with_head:
            return
        if self.head_dev == self.dev0:
            self.h_host.copy_(self.h1)               # dev1 -> host -> dev0
            if _dbg: print("[head] relay h1->host ok", flush=True)
            self.h0.copy_(self.h_host)
            if _dbg: print("[head] relay host->h0 ok", flush=True)
        # head 在 dev1 时不需要中转：layers1 跑完后 self.h1 就是最后的隐藏状态
        if self.gh is not None:
            # ★ mixer + lm_head 全在 dev0，可以像 g0 一样捕获（每步省约 5 ms：
            # lm_head 是 16 个 chunk 的反量化+matmul，eager 下每 token 都要重新发射一遍）。
            self.gh.replay()
        else:
            self._head_device()
        if _dbg: print("[head] after lm_head ok", flush=True)
        if _fp:
            # 数值指纹：模型输出本身是退化的（恒为 '!'），靠文本比对无法判断图捕获有没有
            # 漏掉裸 acl 发射。head() 是 eager 不进图，所以这里的打印不会破坏捕获。
            lg = self.logits.float()
            print(f"[head-fp] sum={lg.sum().item():.6f} amax={lg.amax().item():.6f} "
                  f"argmax={int(lg.argmax().item())} first4={lg.view(-1)[:4].tolist()}", flush=True)

    # ---------------- 图捕获一致性自检（FNX_GRAPH_CHECK=1 时在装配期跑一次） ----------------
    def _state_tensors(self):
        """前向里会被**原地改写**的状态张量（KV 缓存 / PLE 卷积滑窗 / DeltaNet 的 S）。"""
        ts = [t for t in (getattr(self, "ple_conv", None),) if t is not None]
        for kv in (self.kv0, self.kv1):
            ts += list(kv.k) + list(kv.v)
        for L in self.layers0 + self.layers1:
            if getattr(L, "S", None) is not None:
                ts.append(L.S)
            ts.append(L.conv)
            if getattr(L, "ple_conv", None) is not None:
                ts.append(L.ple_conv)
            if getattr(L, "index_k", None) is not None:
                ts.append(L.index_k)
            if getattr(L, "index_keys", None) is not None:
                ts.append(L.index_keys)
        return ts

    def _snap(self):
        return [t.clone() for t in self._state_tensors()]

    def _restore(self, snap):
        for t, s in zip(self._state_tensors(), snap):
            t.copy_(s)

    def _graph_check(self):
        """同一初始状态，eager 一遍 vs 图 replay 一遍，比较 h1。

        ★ 这是唯一能证伪"裸 acl 发射没被记进图"的办法：模型输出是退化的（恒为 '!'），
        看图面数字永远是 200。图里丢了算子不会报错，只会读到没被写过的内存。
        """
        snap = self._snap()
        self.body0()
        if self.layers1:
            self.body1()
        torch.npu.synchronize()
        ref = self.h1.clone()
        self._restore(snap)
        self.g0.replay()
        if self.g1 is not None:
            self.g1.replay()
        torch.npu.synchronize()
        got = self.h1.clone()
        self._restore(snap)
        d = (got.float() - ref.float()).abs()
        base = ref.float().abs().mean() + 1e-9
        rel = float(d.mean() / base)
        verdict = "PASS" if rel < 1e-3 else "FAIL(图里很可能丢了算子)"
        print(f"[graph-check] h1 maxabs={d.max().item():.3e} mean={d.mean().item():.3e} "
              f"rel={rel:.3e} -> {verdict}", flush=True)
        return rel

    def capture(self):
        s = torch.npu.Stream()
        self._gs = s
        with torch.npu.stream(s):
            for _ in range(3):
                self.body0()
                with torch.npu.device(self.dev1):
                    self.body1()
                self.head()
            torch.npu.synchronize()
            self.g0 = torch.npu.NPUGraph()
            with torch.npu.device(self.dev0), torch.npu.graph(self.g0, capture_error_mode="relaxed"):
                self.body0()
            torch.npu.synchronize()
            if E.B == 1 and self.kv0.P > 2048 and any(L.index_enabled for L in self.layers0):
                self.pos0.zero_()
                for L in self.layers0:
                    L.short_qsa = True
                self.body0()
                torch.npu.synchronize()
                self.g0_short = torch.npu.NPUGraph()
                try:
                    with torch.npu.device(self.dev0), torch.npu.graph(
                            self.g0_short, pool=self.g0.pool(), capture_error_mode="relaxed"):
                        self.body0()
                finally:
                    for L in self.layers0:
                        L.short_qsa = False
                torch.npu.synchronize()
            if self.layers1 and _CAP_G1 == "layer":
                # ★ 每层一张**单设备**图（只含 dev1 上的这一层）：
                # 绕开"整段 body1 的图跨设备重放不生效"的限制。层间靠固定缓冲区 self.h1 串联。
                with torch.npu.device(self.dev1):
                    s1 = torch.npu.Stream()
                    self._gs1 = s1
                    self._g1_list = []
                    for L in self.layers1:
                        g = torch.npu.NPUGraph()
                        with torch.npu.stream(s1), torch.npu.graph(g, stream=s1, capture_error_mode="relaxed"):
                            self.h1.copy_(L.forward(self.h1, self.pos1))
                        torch.npu.synchronize()
                        self._g1_list.append(g)
                self._log(f"逐层图捕获 {len(self._g1_list)} 张（dev1）")
                torch.npu.synchronize()
            elif self.layers1 and _CAP_G1 == "1":
                # ★ 给 dev1 **自己建一条 stream** 再捕获。
                # 之前失败的原因是：虽然切了 `torch.npu.device(dev1)`，但捕获仍在**外层
                # dev0 的 stream** 上进行 → dev1 的内核没被记进图，replay 等于空转。
                with torch.npu.device(self.dev1):
                    s1 = torch.npu.Stream()
                self._gs1 = s1
                with torch.npu.device(self.dev1):
                    self.g1 = torch.npu.NPUGraph()
                    with torch.npu.stream(s1), torch.npu.graph(self.g1, stream=s1, capture_error_mode="relaxed"):
                        self.body1()
                torch.npu.synchronize()
            if self.with_head and _CAP_GH and self.head_dev == self.dev0:
                # ⚠️ 只在 head 位于 dev0 时可捕获：dev1 的 NPUGraph 已被证否（第 29 轮 §19）
                self.gh = torch.npu.NPUGraph()
                with torch.npu.graph(self.gh, capture_error_mode="relaxed"):
                    self._head_device()
                torch.npu.synchronize()
        torch.npu.synchronize()
        if os.environ.get("FNX_GRAPH_CHECK") == "1":
            try:
                self._graph_check()
            except Exception as e:  # noqa: BLE001
                print(f"[graph-check] 自检本身失败: {type(e).__name__}: {e}", flush=True)
        self._log("图捕获完成 " + "/".join(x for x in (
            "body0" if self.g0 else "", "body1" if self.g1 else "", "head" if self.gh else "") if x))

    # ---------------- ngram id（宿主整数运算，int64 回绕对齐 torch） ----------------
    def ngram_ids(self, tok):
        hist = self.ctx + [tok]
        prev_eos = -1
        for i in (0, 1):
            if hist[i] == E.NG_EOS:
                prev_eos = i
        pin = 2 - (prev_eos + 1)
        sh = [hist[2 - s] if ((pin >= s) and (2 - s >= 0)) else E.NG_EOS for s in range(3)]
        ids = []
        for ngram in (2, 3):
            m = _s64(sh[0] * self.mult[0])
            for p in range(1, ngram):
                m = _s64((m & MASK64) ^ (_s64(sh[p] * self.mult[p]) & MASK64))
            for h in range((ngram - 2) * 8, (ngram - 1) * 8):
                ids.append(m % self.head_vocab[h] + self.head_off[h])
        return ids

    def embed_rows(self, toks):
        """宿主侧取 len(toks) 行（每行 2560 个 int8）→ 反量化 → 送 n×5 KB 上卡。"""
        n = len(toks)
        idx = toks[0] if n == 1 else torch.tensor(toks, dtype=torch.long)
        e = self.emb_q[idx].view(n, E.H).to(torch.bfloat16)
        s = self.emb_s[idx].view(n, -1).to(torch.bfloat16)
        gs = E.H // s.shape[-1]
        return (e.view(n, -1, gs) * s.unsqueeze(-1)).reshape(n, E.H)

    def embed_row(self, tok):
        return self.embed_rows([int(tok)])

    def step_pad(self, tok, with_head=True):
        """★ 单序列路径在 B>1 模型上的兼容入口：把同一 token 复制到所有行。
        行与行之间完全独立（各自的 S/conv/kv/pos），所以第 0 行的结果与 B=1 时一致，
        读数只看 `row_logits()`。这样 /fnx/loss、/fnx/selfcheck 和未批处理的 generate
        在 B>1 服务里也能照常工作（只是没有提速）。"""
        t = int(tok)
        return self.step([t] * E.B, with_head=with_head)

    def row_logits(self, i=0):
        """第 i 行的 logits（B=1 时就是唯一那行，返回一维）。"""
        return self.logits[i] if E.B > 1 else self.logits.view(-1)

    @torch.no_grad()
    def zero_row(self, r):
        """★ 只清第 r 行的状态（连续批处理用）：新请求加入时占这一行，
        其它行（可能在解码中）完全不受影响。"""
        for L in self.layers0 + self.layers1:
            if hasattr(L, "S"):
                L.S[r].zero_()
            if getattr(L, "index_k", None) is not None:
                L.index_k.zero_()
            if getattr(L, "index_keys", None) is not None:
                L.index_keys.zero_()
            L.conv[r].zero_()
            if getattr(L, "use_ple", False):
                L.ple_conv[r].zero_()
        for st in (self.kv0, self.kv1):
            for t in st.k:
                t[r].zero_()
            for t in st.v:
                t[r].zero_()
        self.pos0[r] = 0
        self.pos1[r] = 0
        self._host_pos = 0
        self.h0[r].zero_()
        self.h1[r].zero_()
        self.emb_buf[r].zero_()
        torch.npu.synchronize()

    def _chunk_body0(self):
        h = self.emb_chunk.repeat(1, 1, E.HC)
        for layer in self.layers0:
            h = layer.forward_chunk(h, self.pos0, E._CHUNK)
        self.chunk_h0.copy_(h)


    @torch.no_grad()
    def capture_chunk(self):
        """Capture several fixed sizes so prompt tails can also use graphs."""
        maximum = E._CHUNK
        sizes = {maximum}
        sizes.update(int(s) for s in os.environ.get("FNX_CHUNK_SIZES", str(maximum)).split(","))
        sizes = sorted((s for s in sizes if 1 < s <= maximum), reverse=True)
        base = self.emb_chunk
        fields = ("gc", "gc_short", "chunk_h0")
        self.chunk_graphs = {}
        try:
            for size in sizes:
                E._CHUNK = size
                self.emb_chunk = base[:size]
                self._capture_chunk_size()
                self.chunk_graphs[size] = {f: getattr(self, f, None) for f in fields}
        finally:
            E._CHUNK = maximum
            self.emb_chunk = base
        for name, value in self.chunk_graphs[maximum].items():
            setattr(self, name, value)

    @torch.no_grad()
    def _capture_chunk_size(self):
        """Capture fixed-size prefill on dev0, sharing decode's memory pool.

        Replay is serialized with decode. dev1 stays eager, as in production.
        Caller must reset/restore model state after capture.
        """
        if E.B != 1 or E._CHUNK <= 1:
            raise ValueError("chunk capture requires FNX_BATCH=1 and FNX_CHUNK>1")
        self.chunk_h0 = torch.zeros(E._CHUNK, 1, E.HCH, dtype=torch.bfloat16,
                                    device=self.dev0)
        self.emb_chunk.zero_()
        self.pos0.zero_()
        self.gc = None
        with torch.npu.device(self.dev0):
            stream = torch.npu.Stream()
            self._chunk_stream = stream
            with torch.npu.stream(stream):
                for _ in range(2):
                    self._chunk_body0()
                torch.npu.synchronize(self.dev0)
                graph = torch.npu.NPUGraph()
                pool = self.g0.pool() if self.g0 is not None and os.environ.get("FNX_CHUNK_POOL", "shared") == "shared" else None
                with torch.npu.graph(graph, stream=stream, pool=pool, capture_error_mode="relaxed"):
                    self._chunk_body0()
                self.gc_short = None
                if self.g0_short is not None and E._CHUNK <= 2048:
                    for L in self.layers0:
                        L.short_qsa = True
                    try:
                        self._chunk_body0()
                        torch.npu.synchronize(self.dev0)
                        self.gc_short = torch.npu.NPUGraph()
                        with torch.npu.graph(self.gc_short, stream=stream, pool=pool, capture_error_mode="relaxed"):
                            self._chunk_body0()
                    finally:
                        for L in self.layers0:
                            L.short_qsa = False
            torch.npu.synchronize(self.dev0)
            self.gc = graph
        self.reset()

    @torch.no_grad()
    def prefill_chunk(self, ids):
        """★ 第 46 轮：分块 prefill（一次算 T 个 token，eager）。

        为什么值得：prefill 原来逐 token 跑同一套 forward，每 token 都要把整层权重读一遍。
        第 45 轮实测（真实 L00 权重）：稠密 GEMM T=32 时 32.7×、MoE 权重 31.1×。
        分块只批量化"与时间无关"的部分（投影/GEMM/KV 行写/注意力），顺序依赖的
        DeltaNet 递推仍逐 token。

        语义：**只推进状态**（KV/conv/S/pos/ctx），不返回 logits —— 最后一个 prompt token
        由调用方用正常的 `step()` 走（它会产出 logits）。
        """
        T = E._CHUNK
        if E.B != 1 or T < 1:
            raise ValueError("prefill_chunk requires FNX_BATCH=1 and a positive chunk size")
        pos0 = int(self.pos0[0])
        n = len(ids)
        if pos0 + n > self.kv0.P:
            raise ValueError("prefill exceeds allocated context")
        i = 0
        if self.emb_chunk.shape[0] < min(T, n):
            # 装载期按 E._CHUNK 分配；测试/服务里改 E._CHUNK 后按需扩容
            self.emb_chunk = torch.zeros(min(T, n), 1, E.H,
                                         dtype=torch.bfloat16, device=self.dev0)
        while i < n:
            m = min(T, n - i)
            bundle = None
            sizes = getattr(self, "chunk_graphs", {})
            eligible = [size for size in sizes if size <= m]
            if eligible:
                m = max(eligible)
                bundle = sizes[m]
            toks = [int(x) for x in ids[i:i + m]]
            if self.emb_chunk.shape[0] < m:
                self.emb_chunk = torch.zeros(m, 1, E.H, dtype=torch.bfloat16,
                                             device=self.dev0)
            buf = self.emb_chunk[:m]
            buf.view(-1).copy_(self.embed_rows(toks).view(-1))
            if bundle is not None or (m == E._CHUNK and getattr(self, "gc", None) is not None):
                with torch.npu.device(self.dev0):
                    active = bundle or {"gc": self.gc, "gc_short": self.gc_short,
                                        "chunk_h0": self.chunk_h0}
                    graph = active["gc_short"] if pos0 + m <= 2048 and active["gc_short"] is not None else active["gc"]
                    graph.replay()
                h = active["chunk_h0"]
            else:
                h = buf.repeat(1, 1, E.HC)                     # [m,1,HCH]
                for L in self.layers0:
                    h = L.forward_chunk(h, pos0, m)
            self.prefill_hidden = h
            if self.layers1:
                # 跨卡中转：dev0 -> host -> dev1（每块一次，不再每 token 一次）
                hh = h.cpu().to(self.dev1)
                with torch.npu.device(self.dev1):
                    for L in self.layers1:
                        L.short_qsa = pos0 + m <= 2048 and L.index_enabled
                        hh = L.forward_chunk(hh, pos0, m)
            self.pos0.add_(m)
            self.pos1.add_(m)
            self._host_pos += m
            self.ctx = (self.ctx + toks)[-len(self.ctx):]
            pos0 += m
            i += m
        torch.npu.synchronize()

    def step(self, tok, with_head=True):
        """tok 可以是单个 int（B=1）或长度为 B 的列表（多请求批处理）。

        多序列批处理的意义见 `engine.B`：微基准证明 int8 权重的单次调用耗时与 B 无关，
        所以一次前向处理 B 条序列可以把"反量化整层权重"的成本摊到 1/B。

        ★ 跨卡 + 图捕获时，中转点必须显式同步。
        现象（实测，16 层 8+8）：eager 与"层主序"逐位一致（0.000%），但图 replay 差 54.169%；
        单卡 16 层三者完全一致（0.000%）。所以问题只出在**跨卡中转 + 图 replay** 的组合：
        `self.h1.copy_(self.h_host)`（H2D 到 dev1）与 `self.g1.replay()`（dev1 的图）
        之间没有顺序保证，dev1 的图会读到**上一个 token 的陈旧 h1**。
        """
        toks = [int(tok)] if isinstance(tok, int) else [int(x) for x in tok]
        if len(toks) != E.B:
            raise ValueError(f"batch {len(toks)} != FNX_BATCH {E.B}")
        if E.B == 1:
            self.tok.fill_(toks[0])
        else:
            self.tok.copy_(torch.tensor(toks, dtype=torch.int64, device=self.tok.device))
        self.emb_buf.view(-1).copy_(self.embed_rows(toks).view(-1))
        if self.table is not None:
            self.ngram_emb.copy_(self.table.gather(self.ngram_ids(toks[0])))
        short = E.B == 1 and self._host_pos < 2048
        for L in self.layers1:
            L.short_qsa = short and L.index_enabled
        if self.g0 is not None:
            graph = self.g0_short if short and self.g0_short is not None else self.g0
            graph.replay()
        else:
            self.body0()
        if self.layers1 or self.with_head:
            self.h_host.copy_(self.h0)
            self.h1.copy_(self.h_host)
        if self.layers1:
            if self._g1_list:
                # replay 走**默认流**：中转拷贝 `h1.copy_(h_host)` 是在默认流上做的，
                # 若把 replay 发到捕获用的旁路流上，两者没有顺序 → replay 读到陈旧 h1
                # （实测：旁路流 replay 时 logits 差 65%，默认流则一致）。
                with torch.npu.device(self.h1.device):
                    for g in self._g1_list:
                        g.replay()
                torch.npu.synchronize(self.h1.device)
            elif self.g1 is not None:
                with torch.npu.device(self.h1.device), torch.npu.stream(self._gs1):
                    self.g1.replay()
                torch.npu.synchronize(self.h1.device)
            else:
                self.body1()
        if with_head:
            if self.gh is not None:
                self.gh.replay()
            else:
                self.head()
        self.pos0.add_(1)
        self.pos1.add_(1)
        self._host_pos += 1
        if E.B == 1:
            self.ctx = self.ctx[1:] + [toks[0]]

    def reset(self):
        for L in self.layers0 + self.layers1:
            if hasattr(L, "S"):
                L.S.zero_()
            L.conv.zero_()
            if getattr(L, "index_k", None) is not None:
                L.index_k.zero_()
            if getattr(L, "index_keys", None) is not None:
                L.index_keys.zero_()
            if L.use_ple:
                L.ple_conv.zero_()
        for st in (self.kv0, self.kv1):
            for t in st.k:
                t.zero_()
            for t in st.v:
                t.zero_()
        self.pos0.zero_()
        self.pos1.zero_()
        self._host_pos = 0
        self.ctx = [E.NG_EOS] * E.NG_CTX
        torch.npu.synchronize()

    @torch.no_grad()
    def save_state(self, to_host=False, compact=None):
        """★ 递归状态快照：把 reset() 会清零的全部状态克隆一份。

        用途 = 前缀缓存。实测（第 30 轮）：聊天模板的静态前缀（system 提示 + 脚手架）
        固定 45 token，每 token 127 ms ⇒ 每个请求光重复 prefill 这段就花 5.71 s，
        而「你好」总共只有 53 token。把这段前缀跑一次、快照状态，之后每请求
        load_state 覆盖（~ms 级）即可省掉这 45 步。
        """
        copy = (lambda t: t.cpu()) if to_host else (lambda t: t.clone())
        if compact is None:
            compact = to_host and E.B == 1 and os.environ.get("FNX_COMPACT_STATE", "1") == "1"
        compact = compact and E.B == 1
        n = int(self.pos0[0]) if compact else self.P
        kv_copy = lambda t: copy(t[:, :, :n])
        st = {"pos0": copy(self.pos0), "pos1": copy(self.pos1),
              "ctx": list(self.ctx), "layers": [],
              "kv0": ([kv_copy(t) for t in self.kv0.k], [kv_copy(t) for t in self.kv0.v]),
              "kv1": ([kv_copy(t) for t in self.kv1.k], [kv_copy(t) for t in self.kv1.v])}
        if compact:
            st["valid_len"] = n
        for L in self.layers0 + self.layers1:
            d = {"conv": copy(L.conv)}
            if getattr(L, "index_k", None) is not None:
                d["index_k"] = copy(L.index_k[:n])
            if getattr(L, "index_keys", None) is not None:
                d["index_keys"] = copy(L.index_keys[:(n + 3) // 4])
            if hasattr(L, "S"):
                d["S"] = copy(L.S)
            if getattr(L, "use_ple", False):
                d["ple_conv"] = copy(L.ple_conv)
            st["layers"].append(d)
        return st

    @torch.no_grad()
    def load_state(self, st):
        """恢复 save_state() 的快照。不先 reset：直接覆盖更快，且语义完全等价。"""
        self.pos0.copy_(st["pos0"])
        self.pos1.copy_(st["pos1"])
        self._host_pos = int(st["pos0"][0])
        self.ctx = list(st["ctx"])
        n = st.get("valid_len", self.P)
        for L, d in zip(self.layers0 + self.layers1, st["layers"]):
            if "index_k" in d:
                L.index_k[:n].copy_(d["index_k"])
            if "index_keys" in d:
                L.index_keys[:(n + 3) // 4].copy_(d["index_keys"])
            if "S" in d:
                L.S.copy_(d["S"])
            L.conv.copy_(d["conv"])
            if "ple_conv" in d:
                L.ple_conv.copy_(d["ple_conv"])
        for kv, (sk, sv) in ((self.kv0, st["kv0"]), (self.kv1, st["kv1"])):
            for t, s in zip(kv.k, sk):
                t[:, :, :n].copy_(s)
            for t, s in zip(kv.v, sv):
                t[:, :, :n].copy_(s)
        torch.npu.synchronize()

    @torch.no_grad()
    def save_into(self, st):
        """把当前状态就地写入一个已分配好的快照槽（不 alloc，避免显存碎片）。

        实测教训：每请求 `save_state()` clone 100 多块不等的张量会让 HBM 占用
        从 58.9 GB 涨到 63.6 GB（碎片），所以槽位只在首次分配，之后一律 copy_ 复用。
        """
        if "valid_len" in st and st["valid_len"] != int(self.pos0[0]):
            st.update(self.save_state(to_host=st["pos0"].device.type == "cpu", compact=True))
            return
        n = st.get("valid_len", self.P)
        st["pos0"].copy_(self.pos0)
        st["pos1"].copy_(self.pos1)
        st["ctx"] = list(self.ctx)
        for L, d in zip(self.layers0 + self.layers1, st["layers"]):
            if "S" in d:
                d["S"].copy_(L.S)
            if "index_k" in d:
                d["index_k"].copy_(L.index_k[:n])
            if "index_keys" in d:
                d["index_keys"].copy_(L.index_keys[:(n + 3) // 4])
            d["conv"].copy_(L.conv)
            if "ple_conv" in d:
                d["ple_conv"].copy_(L.ple_conv)
        for kv, (sk, sv) in ((self.kv0, st["kv0"]), (self.kv1, st["kv1"])):
            for t, s in zip(kv.k, sk):
                s.copy_(t[:, :, :n])
            for t, s in zip(kv.v, sv):
                s.copy_(t[:, :, :n])
        torch.npu.synchronize()

    @staticmethod
    def state_to_host(st):
        """快照搬到主机内存。★ 必须这样做：单卡已用到 62.1/64 GiB，只剩 1.9 GiB，
        而前缀快照 152 MiB × 3 个槽就吃掉 456 MiB。搬到主机后每请求多花 ~10 ms
        （152 MiB H2D @ ~21 GB/s），换掉 OOM 风险。"""
        def conv(v):
            if torch.is_tensor(v):
                return v.to("cpu")
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            if isinstance(v, list):
                return [conv(x) for x in v]
            if isinstance(v, tuple):
                return tuple(conv(x) for x in v)
            return v
        return conv(st)

    def state_bytes(self, st):
        n = 0

        def acc(v):
            nonlocal n
            if torch.is_tensor(v):
                n += v.numel() * v.element_size()
            elif isinstance(v, (list, tuple)):
                for x in v:
                    acc(x)
            elif isinstance(v, dict):
                for x in v.values():
                    acc(x)
        acc(st)
        return n


@torch.no_grad()
def gen_greedy(m, ids, max_new=8):
    m.reset()
    for t in ids[:-1]:
        m.step_pad(int(t), with_head=False)
    nxt = int(ids[-1])
    out = []
    for _ in range(max_new):
        m.step_pad(nxt, with_head=True)
        nxt = int(m.logits.float().argmax(dim=-1).item())
        out.append(nxt)
        if nxt == E.NG_EOS:
            break
    return out
