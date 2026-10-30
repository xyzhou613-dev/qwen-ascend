"""步骤 (f)：OpenAI 兼容服务（batch=1 专用后端，每请求重置 DeltaNet 递归状态）。

启动：
  docker exec -d fnx bash -lc "cd /root/ascend-dev/fnx && /root/ascend-dev/venv/bin/python -m fnx.server"
接口：
  GET  /health            GET /v1/models
  POST /v1/chat/completions   （支持 stream=true 的 SSE）
"""
import asyncio
import json
import os
import re
import threading
import time
import uuid

import torch

from . import engine as E
from .model import FnModel

MODEL_ID = "qwen3.8-flash-next-w4"
P = int(os.environ.get("FNX_CTX", "2048"))
# ★ 每请求分段计时开关：开时在 generate() 里插 torch.npu.synchronize() 才能拿到
#   设备真实耗时（否则 replay 只是入队）。生产建议保持 0，排查时临时设 1。
_PROF = os.environ.get("FNX_PROF", "0") == "1"
PROF_LOG = []          # 最近若干请求的分段耗时
_PROF_MAX = 32

# ★ 输出尾部清理：思考关掉后模型有时会越过轮尾继续"编造下一轮模板"
#   （实测尾部出现 "\nuser\nassistant\n<think>"）。只在**行首角色名/模板标记**处裁，
#   避免误伤正文里出现的 assistant / system 等单词。
_TAIL_RE = re.compile(
    r"(?:\s*(?:<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<think>|</think>)\s*"
    r"|\n\s*(?:user|assistant|system)\s*(?=\n|$))+$")


def clean_text(text):
    return _TAIL_RE.sub("", text) if text else text


# ★ 服务指标（第30轮）：现在所有请求走一把全局锁、且只有 /fnx/prof 的"最后一次"快照，
#   排队情况、累积吞吐、命中率都看不见。这里补一份累积计数器 + GET /metrics。
METRICS = {
    "requests": 0, "requests_err": 0, "tokens_out": 0,
    "prefill_tokens": 0, "prefix_hit_tokens": 0,
    "queue_ms_sum": 0.0, "ttft_ms_sum": 0.0, "tpot_ms_sum": 0.0, "tpot_n": 0,
}


def _metric_add(**kw):
    for k, v in kw.items():
        METRICS[k] = METRICS.get(k, 0) + v

_torch_mod = None


def _import_web():
    global _torch_mod
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse
    import uvicorn
    return FastAPI, Request, JSONResponse, StreamingResponse, uvicorn


FastAPI, Request, JSONResponse, StreamingResponse, uvicorn = _import_web()
app = FastAPI(title="fnx-backend")
LOCK = threading.Lock()
STATE = {}


class Engine:
    def __init__(self):
        t = time.time()
        # ★层数/每卡层数支持环境变量覆盖（默认与原来完全一致，零行为变化）
        #   8 层验证：FNX_LPC=4 FNX_NLAYERS=8
        _lpc = int(os.environ.get("FNX_LPC", "23"))
        _nl = int(os.environ.get("FNX_NLAYERS", "46"))
        self.m = FnModel(P=P, lpc=_lpc, dev0="npu:0", dev1="npu:1", nlayers=_nl,
                         with_head=True, verbose=True)
        print(f"[fnx] 装配 {time.time()-t:.1f}s", flush=True)
        if os.environ.get("FNX_NO_CAPTURE") == "1":
            print("[fnx] FNX_NO_CAPTURE=1 -> 跳过图捕获（全部 eager）", flush=True)
        else:
            t = time.time()
            self.m.capture()
            print(f"[fnx] 图捕获 {time.time()-t:.1f}s", flush=True)
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(
            os.environ.get("FNX_SRCDIR", "/models/Qwen3.8-Flash-Next"))
        # ★停止符（第 30 轮修的真 bug）：`E.NG_EOS`(=248044) 是 **<|endoftext|>/pad**，
        #   而对话轮真正的结束符是 `<|im_end|>`(=248046，也就是 tokenizer 的 eos)。
        #   只停 248044 ⇒ 模型越过轮尾继续编造下一轮模板
        #   （实测尾部出现 "\nuser\nassistant\n<think>"）。这里把所有轮尾/文本尾都加上。
        self.stop_ids = {int(E.NG_EOS)}
        if self.tok.eos_token_id is not None:
            self.stop_ids.add(int(self.tok.eos_token_id))
        for _t in ("<|im_end|>", "<|endoftext|>"):
            _i = self.tok.convert_tokens_to_ids(_t)
            if isinstance(_i, int) and 0 <= _i < len(self.tok):
                self.stop_ids.add(int(_i))
        print(f"[fnx] 停止符 = {sorted(self.stop_ids)}", flush=True)
        self.n_req = 0
        self.last_prof = {}
        self.last_tok = {}
        self.last_sse = {}
        self.prefix_ids = []
        self.prefix_state = None
        # ★ 模板变体：reasoning_effort ∈ {xhigh, medium, low} + enable_thinking 开关。
        #   实测静态前缀长度：xhigh=45 token / low=33 / medium=3 / 关思考=3。
        #   每个变体都要有自己的前缀状态（前缀按变体缓存，互不通用）。
        self.prefix_variants = {}
        self.tpl_default = self._tpl_kw({})
        # ★ 对话级前缀缓存：多轮对话每轮都会把整段历史重跑一遍（第30轮实测：第3轮
        #   67 步 = 8.6 s）。能复用的是 **prefill 结束点** 的状态（键 = prompt[:-1]），
        #   它逐位等于下一轮 prompt 的历史部分（第30轮 §4 + turn_diag.py 实测公共前缀
        #   恰好 == key 全长）。
        #   ⚠️ 早先版本用"新 key 比最短的还短就跳过"的策略 ⇒ 第一轮 12 token 的 key
        #   永远进不了槽，导致**第 2 轮必然 miss**。改成 LRU + 4 槽。
        self.conv_keys = []
        self.conv_slots = []
        self.conv_used = []
        self.conv_tick = 0
        self.conv_n = int(os.environ.get("FNX_CONV_CACHE", "4"))
        if os.environ.get("FNX_PREFIX_CACHE", "1") == "1":
            try:
                self._build_prefix(self.tpl_default)
            except Exception as ex:      # 前缀缓存失败绝不能挡住服务
                print(f"[fnx] 前缀缓存失败（关闭）: {ex!r}", flush=True)
        for _d in (0, 1):
            try:
                print(f"[fnx] hbm{_d}: alloc={torch.npu.memory_allocated(_d)/2**30:.2f} GiB "
                      f"reserved={torch.npu.memory_reserved(_d)/2**30:.2f} GiB", flush=True)
            except Exception:
                pass

    @staticmethod
    def _tpl_kw(body):
        """每请求的模板参数：reasoning_effort / enable_thinking（body 优先，其次环境变量）。"""
        kw = {}
        eff = body.get("reasoning_effort") or os.environ.get("FNX_EFFORT", "")
        if eff:
            kw["reasoning_effort"] = str(eff)
        et = body.get("enable_thinking")
        if et is None:
            _e = os.environ.get("FNX_ENABLE_THINKING", "")
            if _e != "":
                et = _e not in ("0", "false", "False", "")
        if et is not None:
            kw["enable_thinking"] = bool(et)
        return kw

    @staticmethod
    def _tpl_key(kw):
        return json.dumps(kw, sort_keys=True, ensure_ascii=False)

    def _build_prefix(self, kw):
        """★ 前缀缓存：模板静态前缀（第一条 user 内容之前的所有 token）只跑一次。

        第 30 轮实测根因：聊天模板带一段固定 system 提示，xhigh 下静态前缀 = 45 token，
        每 token ~130 ms ⇒ 每请求光重复 prefill 就 5.7 s，而「你好」全文才 53 token。
        这里用哨兵串渲染模板切出前缀，跑一遍后快照全部递归状态；之后每请求
        load_state 覆盖（ms 级）代替 reset+前缀 prefill。
        """
        key = self._tpl_key(kw)
        if key in self.prefix_variants:
            return self.prefix_variants[key]
        # 变体表上限：每个变体占 151.9 MiB 主机内存，不能被客户端用任意 kwargs 撑爆
        if len(self.prefix_variants) >= int(os.environ.get("FNX_PREFIX_MAX", "4")):
            print(f"[fnx] 前缀变体已达上限，{key} 走全量 prefill", flush=True)
            return [], None
        SENT = "\u0002FNXPFX\u0002"
        text = self.tok.apply_chat_template(
            [{"role": "user", "content": SENT}], tokenize=False,
            add_generation_prompt=True, **kw)
        cut = text.find(SENT)
        if cut <= 0:
            print("[fnx] 前缀缓存：模板里找不到哨兵 -> 关闭", flush=True)
            self.prefix_variants[key] = ([], None)
            return [], None
        pids = [int(x) for x in
                self.tok(text[:cut], return_tensors="pt").input_ids[0].tolist()]
        t = time.time()
        if pids:
            self.m.reset()
            for x in pids:
                self.m.step_pad(x, with_head=False)
            st = self.m.state_to_host(self.m.save_state())
            mb = self.m.state_bytes(st) / 2 ** 20
        else:
            st, mb = None, 0.0
        self.prefix_variants[key] = (pids, st)
        if os.environ.get("FNX_PREFIX_CACHE", "1") == "1":
            self.prefix_ids, self.prefix_state = pids, st
        print(f"[fnx] 前缀缓存[{key}] {len(pids)} token, 状态 {mb:.1f} MiB(主机), "
              f"建缓存 {time.time()-t:.1f}s", flush=True)
        return pids, st

    def _prefix_hit(self, prompt_ids, tpl_kw):
        """返回可复用前缀的 (token 数, 状态)；逐位相等才用，否则全量 prefill。

        候选 = 该模板变体的静态前缀 + 最近几轮 prefill 结束点状态，取最长匹配。
        """
        best_n, best_st = 0, None
        if os.environ.get("FNX_PREFIX_CACHE", "1") == "1":
            pids, pst = self._build_prefix(tpl_kw)
            if pids and pst is not None and len(prompt_ids) > len(pids) \
                    and prompt_ids[:len(pids)] == pids:
                best_n, best_st = len(pids), pst
        for i, (key, st) in enumerate(zip(self.conv_keys, self.conv_slots)):
            k = len(key)
            if k > best_n and len(prompt_ids) > k and tuple(prompt_ids[:k]) == key:
                best_n, best_st = k, st
                self.conv_tick += 1
                self.conv_used[i] = self.conv_tick
        return best_n, best_st

    def _conv_store(self, prompt_ids, n0):
        """把 prefill 结束点的状态存进槽位（键 = prompt[:-1]，与状态严格对应）。

        prompt[:-1] 是被喂进模型的那一段（最后一个 prompt token 由 decode 首步喂），
        它逐位等于下一轮 prompt 的历史部分，所以下一轮可直接 load_state。
        """
        if self.conv_n <= 0:
            return
        k = len(prompt_ids) - 1
        if k - n0 < 4:                      # 省不到 4 个 token 就不占槽
            return
        key = tuple(int(x) for x in prompt_ids[:k])
        if key in self.conv_keys:
            return
        m = self.m
        if len(self.conv_slots) < self.conv_n:
            self.conv_slots.append(self.m.state_to_host(m.save_state()))
            self.conv_keys.append(key)
            self.conv_tick += 1
            self.conv_used.append(self.conv_tick)
            return
        # LRU 淘汰：聊天里历史会越来越长，必须保住"最近几轮"这条链，
        # 不能按长度排序（那样第一轮 12 token 的 key 永远进不来）。
        i = min(range(len(self.conv_used)), key=lambda j: self.conv_used[j])
        m.save_into(self.conv_slots[i])
        self.conv_keys[i] = key
        self.conv_tick += 1
        self.conv_used[i] = self.conv_tick

    @torch.no_grad()
    def generate(self, prompt_ids, max_new, temperature=0.0, tpl_kw=None):
        """每请求必须 reset：replay 会原地更新 ΔNet 递归状态与 KV，不重置会发散 NaN。"""
        m = self.m
        tpl_kw = tpl_kw or {}
        prof = {"n_prompt": len(prompt_ids), "max_new": max_new}
        synth = torch.npu.synchronize if _PROF else None
        n0, st0 = self._prefix_hit(prompt_ids, tpl_kw)
        prof["prefix_hit"] = n0

        if synth:
            synth()
            t_a = time.perf_counter()
        if n0:
            m.load_state(st0)
        else:
            m.reset()
        if synth:
            synth()
            t_b = time.perf_counter()
            prof["reset_ms"] = round((t_b - t_a) * 1e3, 2)
        else:
            t_b = time.perf_counter()
        m.id = uuid.uuid4().hex
        # ★ prefill 进度：单 token 约 143 ms，长 prompt 要几十秒却"静默"。
        #   这里按 1/8 刻度发进度事件（流式时变成 SSE 注释行，客户端可显示"正在读 prompt"）。
        pre = prompt_ids[n0:-1]
        n_pre = len(pre)
        every = max(1, n_pre // 8)
        for i, x in enumerate(pre):
            m.step_pad(int(x), with_head=False)
            if n_pre >= 16 and (i + 1) % every == 0:
                yield ("p", i + 1, n_pre)
        if synth:
            synth()
        t_c = time.perf_counter()
        if synth:
            # 只有开 PROF 时才有设备真值；否则这里只是入队耗时，标出来会误导
            prof["prefill_ms"] = round((t_c - t_b) * 1e3, 2)
        prof["n_prefill"] = n_pre
        try:
            self._conv_store(prompt_ids, n0)      # 缓存失败绝不能挡住服务
        except Exception as ex:
            print(f"[fnx] 对话缓存失败（忽略）: {ex!r}", flush=True)
        if n_pre:
            yield ("p", n_pre, n_pre)
        nxt = int(prompt_ids[-1])
        out = []
        step_ms, samp_ms = [], []
        for _ in range(max_new):
            if synth:
                synth()
                t_d = time.perf_counter()
            m.step_pad(nxt, with_head=True)
            if synth:
                synth()
                t_e = time.perf_counter()
            logits = m.row_logits().float().view(-1)
            if temperature <= 0:
                nxt = int(logits.argmax(dim=-1).item())
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                nxt = int(torch.multinomial(probs, 1).item())
            if synth:
                t_f = time.perf_counter()
                step_ms.append(round((t_e - t_d) * 1e3, 2))
                samp_ms.append(round((t_f - t_e) * 1e3, 2))
            out.append(nxt)
            yield ("t", nxt)
            if nxt in self.stop_ids:
                break
        if synth and step_ms:
            prof["steps"] = len(step_ms)
            prof["step_ms_avg"] = round(sum(step_ms) / len(step_ms), 2)
            prof["step_ms_first"] = step_ms[0]
            prof["samp_ms_avg"] = round(sum(samp_ms) / len(samp_ms), 2)
        # 快照已在 prefill 结束时存（见上方 _conv_store），这里只记录分段耗时
        self.last_prof = prof
        PROF_LOG.append(prof)
        del PROF_LOG[:-_PROF_MAX]


def get_engine():
    with LOCK:
        if "e" not in STATE:
            STATE["e"] = Engine()
        return STATE["e"]


@app.get("/metrics")
def metrics():
    """★ 累积指标（Prometheus 文本格式）。工程化用：排队/吞吐/命中率/延迟。"""
    m = METRICS
    n = max(1, int(m["requests"]))
    lines = [
        "# TYPE fnx_requests_total counter",
        f"fnx_requests_total {m['requests']}",
        f"fnx_requests_err_total {m['requests_err']}",
        f"fnx_tokens_out_total {m['tokens_out']}",
        f"fnx_prefill_tokens_total {m['prefill_tokens']}",
        f"fnx_prefix_hit_tokens_total {m['prefix_hit_tokens']}",
        "# TYPE fnx_queue_ms_avg gauge",
        f"fnx_queue_ms_avg {m['queue_ms_sum'] / n:.3f}",
        f"fnx_ttft_ms_avg {m['ttft_ms_sum'] / n:.3f}",
        f"fnx_tpot_ms_avg {m['tpot_ms_sum'] / max(1, m['tpot_n']):.3f}",
        f"fnx_prefill_hit_ratio {m['prefix_hit_tokens'] / max(1, m['prefix_hit_tokens'] + m['prefill_tokens']):.4f}",
        f"fnx_engine_loaded {1 if 'e' in STATE else 0}",
    ]
    return StreamingResponse(iter(["\n".join(lines) + "\n"]), media_type="text/plain")


@app.get("/health")
def health():
    return {"status": "ok", "engine_loaded": "e" in STATE, "ctx": P}


@app.get("/fnx/prof")
def prof():
    """★ 每请求分段耗时（查找 ~5.7 s 固定开销用）。FNX_PROF=1 才含设备侧真值。"""
    e = STATE.get("e")
    if e is None:
        return {"engine_loaded": False}
    return {"engine_loaded": True, "prof_on": _PROF,
            "tpl_default": e.tpl_default,
            "prefix_variants": {k: len(v[0]) for k, v in e.prefix_variants.items()},
            "conv_keys": [len(k) for k in e.conv_keys],
            "last_tok": e.last_tok, "last_prof": e.last_prof,
            "last_sse": getattr(e, "last_sse", {}), "log": PROF_LOG}


@app.get("/v1/models")
def models():
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model", "owned_by": "fnx"}]}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    eng = get_engine()
    msgs = body.get("messages", [])
    max_new = int(body.get("max_tokens") or 128)
    temp = float(body.get("temperature") or 0.0)
    stream = bool(body.get("stream"))

    t_t0 = time.perf_counter()
    tpl_kw = Engine._tpl_kw(body)
    text = eng.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                       **tpl_kw)
    t_tk = time.perf_counter()
    ids = eng.tok(text, return_tensors="pt").input_ids[0].tolist()
    eng.last_tok = {"chat_template_ms": round((t_tk - t_t0) * 1e3, 2),
                    "tokenize_ms": round((time.perf_counter() - t_tk) * 1e3, 2),
                    "n_prompt": len(ids), "tpl": tpl_kw}

    if len(ids) > P:
        _metric_add(requests_err=1)
        return JSONResponse(status_code=400, content={
            "error": f"prompt {len(ids)} tokens > ctx {P}"})
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    t0 = time.perf_counter()
    t_q = time.perf_counter()
    eng.n_req += 1
    _metric_add(requests=1)

    if not stream:
        # ★ 必须放到线程里跑：`async def` 里做同步生成会**阻塞事件循环**，
        #   生成期间 /health 直接超时 —— 看护就会误判"服务挂了"并重启，
        #   把正在跑的请求连人带服务一起杀（第30轮实测：MCQ 跑到第 23 题被杀）。
        def _work():
            with LOCK:
                qms = (time.perf_counter() - t_q) * 1e3
                toks_ = []
                for ev in eng.generate(ids, max_new, temp, tpl_kw):
                    if ev[0] == "t":
                        toks_.append(ev[1])
                txt_ = clean_text(eng.tok.decode(toks_, skip_special_tokens=True))
                return toks_, txt_, qms, (eng.last_prof or {})

        toks, txt, qms, p = await asyncio.to_thread(_work)
        dt = time.perf_counter() - t0
        _metric_add(queue_ms_sum=qms, tokens_out=len(toks),
                    prefix_hit_tokens=p.get("prefix_hit", 0),
                    prefill_tokens=max(0, p.get("n_prefill", 0)),
                    tpot_ms_sum=dt / max(1, len(toks)) * 1e3, tpot_n=1)
        return {
            "id": cid, "object": "chat.completion", "created": created,
            "model": MODEL_ID,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": txt},
                         "finish_reason": "stop" if (toks and toks[-1] in eng.stop_ids) else "length"}],
            "usage": {"prompt_tokens": len(ids), "completion_tokens": len(toks),
                      "total_tokens": len(ids) + len(toks)},
            "fnx_stats": {"ttft_ms": None, "seconds": round(dt, 3),
                          "tpot_ms": round(dt / max(1, len(toks)) * 1e3, 2),
                          # ★ 让客户端能提前显示"要等多久"：prefill 步数 × 实测 ~143 ms
                          "n_prefill": p.get("n_prefill", 0),
                          "est_prefill_s": round(max(0, p.get("n_prefill", 0)) * 0.143, 2)},
        }

    def sse():
        first = None
        n = 0
        t_dc = 0.0
        # ★ 流式的锁必须持有**整个生成期间**（原来 `with LOCK` 只包住 return，
        #   生成器是之后才被消费的 → 两个流会同时改同一套递归状态/KV）。
        LOCK.acquire()
        try:
            for ev in eng.generate(ids, max_new, temp, tpl_kw):
                if ev[0] == "p":
                    # ★ prefill 进度：SSE 注释行（客户端会忽略，但连接不空闲超时，
                    #   UI 可以据此显示"正在读 prompt i/N"）
                    yield f": prefill {ev[1]}/{ev[2]}\n\n"
                    continue
                t = ev[1]
                if first is None:
                    first = time.perf_counter() - t0
                n += 1
                a = time.perf_counter()
                piece = eng.tok.decode([t], skip_special_tokens=True)
                d = {"id": cid, "object": "chat.completion.chunk", "created": created,
                     "model": MODEL_ID,
                     "choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]}
                s = "data: " + json.dumps(d, ensure_ascii=False) + "\n\n"
                t_dc += time.perf_counter() - a
                yield s
        finally:
            LOCK.release()
        dt = time.perf_counter() - t0
        p0 = eng.last_prof or {}
        done = {"id": cid, "object": "chat.completion.chunk", "created": created,
                "model": MODEL_ID,
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "fnx_stats": {"ttft_ms": round((first or 0) * 1e3, 2), "tokens": n,
                              "seconds": round(dt, 3),
                              "tpot_ms": round((dt - (first or 0)) / max(1, n - 1) * 1e3, 2),
                              "tok_decode_ms": round(t_dc * 1e3, 2),
                              "n_prefill": p0.get("n_prefill", 0)}}
        eng.last_sse = done["fnx_stats"]
        p = eng.last_prof or {}
        _metric_add(tokens_out=n,
                    prefix_hit_tokens=p.get("prefix_hit", 0),
                    prefill_tokens=max(0, p.get("n_prefill", 0)),
                    ttft_ms_sum=(first or 0) * 1e3,
                    tpot_ms_sum=done["fnx_stats"]["tpot_ms"] or 0.0, tpot_n=1)
        yield "data: " + json.dumps(done, ensure_ascii=False) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


@app.post("/fnx/loss")
async def loss(req: Request):
    """★ 质量口径：给定纯文本，teacher-forcing 算每 token 的 NLL。

    为什么需要：这个模型输出退化（token 沙拉），只看 `http=200` 和文本根本判不出好坏，
    也没有历史质量数。NLL/PPL 是可比的连续量，改任何一处（量化组大小、打包方式、层数）
    都能立刻看到代价。
    请求：{"text": "...", "max_tokens": 400}
    返回：{"mean_nll":..., "ppl":..., "n":..., "first_nll":[...], "tokens":[...]}
    """
    import math
    body = await req.json()
    eng = get_engine()
    text = body.get("text") or ""
    cap = int(body.get("max_tokens") or 400)
    ids = eng.tok(text, return_tensors="pt").input_ids[0].tolist()
    if len(ids) < 2:
        return JSONResponse(status_code=400, content={"error": "text 太短"})
    if len(ids) > P:
        ids = ids[:P]
    ids = ids[:cap + 1]
    with LOCK, torch.no_grad():
        m = eng.m
        m.reset()
        m.id = uuid.uuid4().hex
        V = int(m.logits.shape[-1])
        shifts = (-3, -2, -1, 0, 1, 2, 3)
        acc = {d: 0.0 for d in shifts}
        hit = 0
        ranks = []
        nlls = []
        diffs = {}
        for i in range(len(ids) - 1):
            m.step_pad(int(ids[i]), with_head=True)
            lg = m.row_logits().float().view(-1)
            lp = torch.log_softmax(lg, dim=-1)
            tgt = ids[i + 1]
            for d in shifts:                       # ★ 平移扫描：词表是否整体错位
                j = tgt + d
                if 0 <= j < V:
                    acc[d] += float(-lp[j].item())
            if i < 200:
                nlls.append(round(float(-lp[tgt].item()), 4))
                am = int(lg.argmax().item())
                hit += int(am == tgt)
                ranks.append(int((lg > lg[tgt]).sum().item()))
                diffs[am - tgt] = diffs.get(am - tgt, 0) + 1
        n = len(ids) - 1
        top_diffs = sorted(diffs.items(), key=lambda kv: -kv[1])[:5]
        return {"mean_nll": round(acc[0] / n, 6),
                "ppl": round(math.exp(acc[0] / n), 4), "n": n,
                "shift_nll": {str(d): round(acc[d] / n, 4) for d in shifts},
                "argmax_minus_tgt_top5": top_diffs,
                "top1_acc": round(hit / max(1, len(ranks)), 4),
                "median_rank": sorted(ranks)[len(ranks) // 2] if ranks else -1,
                "rank_n": len(ranks),
                "vocab": V,
                "first_nll": [round(float(v), 4) for v in
                              [acc[0] / n]],
                # ★ 逐 token NLL/rank（前 64 个）：与 bf16 参考实现做**配对**比较用
                "nll_tokens": nlls[:64],
                "rank_tokens": ranks[:64],
                "tokens": ids[:24], "prompt_tokens": len(ids)}


@app.post("/fnx/selfcheck")
async def selfcheck(req: Request):
    """★★ 历史上下文自检：模型能不能"复述自己刚说过的话"。

    为什么这是决定性的：单层对拍、端到端逐级对拍全都在 pos=0 做，**从没验证过 KV 缓存/历史**。
    如果模型对自己刚生成的 token 做 teacher-forcing 的 NLL 也很高（≈ln(V)），
    说明它每步都只看到 1 个 token，历史根本没接进去 —— 那才会让任何深度都退化成均匀分布。
    三个口径：
      A 有历史 teacher-force 自己生成的结果   （正常模型应接近 0）
      B 每个位置都把状态重置、只喂 1 个 token（等价于无历史，理论上限）
      C 直接 greedy 生成出来的 token 序列
    """
    import math
    body = await req.json()
    n = int(body.get("n") or 24)
    eng = get_engine()
    tok = eng.tok
    base = tok(body.get("text") or "你好", return_tensors="pt").input_ids[0].tolist()

    with LOCK, torch.no_grad():
        m = eng.m
        # ---- C：先生成 ----
        m.reset()
        m.id = uuid.uuid4().hex
        for x in base[:-1]:
            m.step_pad(int(x), with_head=False)
        seq = []
        cur = int(base[-1])
        for _ in range(n):
            m.step_pad(cur, with_head=True)
            cur = int(m.row_logits().float().argmax(-1).item())
            seq.append(cur)
        full = base[:-1] + seq
        start = len(base) - 1

        # ---- A：有历史 ----
        m.reset()
        m.id = uuid.uuid4().hex
        nll_a = []
        for i, x in enumerate(full[:-1]):
            m.step_pad(int(x), with_head=True)
            if i >= start - 1:
                lp = torch.log_softmax(m.row_logits().float().view(-1), dim=-1)
                nll_a.append(float(-lp[full[i + 1]].item()))

        # ---- B：无历史（每个位置单独喂一个 token）----
        nll_b = []
        for i in range(start, len(full) - 1):
            m.reset()
            m.id = uuid.uuid4().hex
            m.step_pad(int(full[i]), with_head=True)
            lp = torch.log_softmax(m.row_logits().float().view(-1), dim=-1)
            nll_b.append(float(-lp[full[i + 1]].item()))

    V = int(m.logits.shape[-1])
    ma = sum(nll_a) / max(1, len(nll_a))
    mb = sum(nll_b) / max(1, len(nll_b))
    return {"vocab": V, "ln_vocab": round(math.log(V), 4),
            "gen_ids": seq, "gen_text": tok.decode(seq),
            "nll_with_history": round(ma, 4),
            "nll_no_history": round(mb, 4),
            "history_gain": round(mb - ma, 4),
            "verdict": ("历史有效" if ma < mb - 1.0 else
                        "历史几乎没用（每步只看到 1 个 token → KV/state 没接上）")}


if __name__ == "__main__":
    port = int(os.environ.get("FNX_PORT", "8000"))
    print(f"[fnx] 预加载引擎...", flush=True)
    get_engine()
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
