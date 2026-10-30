"""48/48 显存可行性实测（不加载模型，只按真实字节数实分配）。

依据（字节数全部取自 /models/Qwen3.8-Flash-Next-QP/manifest.json 的实测字段）：
  每层 experts p4 : gate_up 512*1280*2560/2 = 838,860,800 B, down 512*2560*640/2 = 419,430,400 B
  每层 experts s4 : gate_up 512*1280*80*2   = 104,857,600 B, down 512*2560*20*2   =  52,428,800 B
  每层 稠密 int8   : 77,300,000 B（第 25 轮 §10.3 实测 77.3 M 参数/层）
  → 每层合计 1,492,878,800 B = 1.3907 GiB；24 层/卡 = 35,829,091,200 B = 33.37 GiB

同时测三件真机上最容易崩的事：
  1) 24 层全部驻留后每卡还剩多少
  2) 第 25 轮 §13.2 触发 OOM 的那次 1.56 GiB 临时分配，现在能不能过
  3) 再叠 lm_head(int8, 0.59 GiB) + 2 GiB workspace 后还剩多少
"""
import torch
import torch_npu  # noqa: F401

G = 1024 ** 3


def gb(n):
    return n / G


# 每层的张量：(名字, 字节数, dtype)
LAYER_TENSORS = [
    ("p4_gate_up", 512 * 1280 * 2560 // 2, torch.int8),
    ("p4_down", 512 * 2560 * 640 // 2, torch.int8),
    ("s4_gate_up", 512 * 1280 * 80 * 2, torch.float16),
    ("s4_down", 512 * 2560 * 20 * 2, torch.float16),
    ("dense_int8", 77_300_000, torch.int8),
]
PER_LAYER = sum(t[1] for t in LAYER_TENSORS)
LAYERS = 24
TEMP = int(1.56 * G)          # §13.2 里触发 OOM 的那次临时分配
LM_HEAD = 636_000_000         # int8 常驻（第 25 轮 §10.3）
WORKSPACE = 2 * G


def probe(dev):
    torch.npu.set_device(dev)
    free0, total = torch.npu.mem_get_info(dev)
    print(f"[dev{dev}] start: free {gb(free0):.2f} / total {gb(total):.2f} GiB", flush=True)
    keep = []
    for li in range(LAYERS):
        for name, nbytes, dt in LAYER_TENSORS:
            keep.append(torch.empty(nbytes, dtype=dt, device=f"npu:{dev}"))
    free1, _ = torch.npu.mem_get_info(dev)
    print(f"[dev{dev}] {LAYERS} 层驻留后（{gb(PER_LAYER * LAYERS):.2f} GiB 逻辑）: "
          f"free {gb(free1):.2f} GiB", flush=True)

    ok_temp = True
    try:
        t = torch.empty(TEMP, dtype=torch.int8, device=f"npu:{dev}")
        del t
    except Exception as e:  # noqa: BLE001
        ok_temp = False
        print(f"[dev{dev}] 1.56 GiB 临时分配失败: {type(e).__name__}", flush=True)
    free2, _ = torch.npu.mem_get_info(dev)
    print(f"[dev{dev}] 1.56 GiB 临时分配 {'成功' if ok_temp else '失败'} → free {gb(free2):.2f} GiB", flush=True)

    extra = []
    try:
        extra.append(torch.empty(LM_HEAD, dtype=torch.int8, device=f"npu:{dev}"))
        extra.append(torch.empty(WORKSPACE, dtype=torch.int8, device=f"npu:{dev}"))
        ok_extra = True
    except Exception as e:  # noqa: BLE001
        ok_extra = False
        print(f"[dev{dev}] lm_head+workspace 分配失败: {type(e).__name__}", flush=True)
    free3, _ = torch.npu.mem_get_info(dev)
    print(f"[dev{dev}] lm_head({gb(LM_HEAD):.2f}) + workspace(2.00) {'成功' if ok_extra else '失败'}"
          f" → free {gb(free3):.2f} GiB", flush=True)

    # 结论
    verdict = "PASS" if (ok_temp and free1 > 0.3 * G) else "FAIL"
    print(f"[dev{dev}] === 48/48 显存可行性: {verdict} "
          f"(24 层后余 {gb(free1):.2f} GiB，临时 1.56 GiB {'OK' if ok_temp else 'NG'}) ===", flush=True)
    return verdict


if __name__ == "__main__":
    print(f"[mem48] 每层 {PER_LAYER} B = {gb(PER_LAYER):.4f} GiB；24 层 = {gb(PER_LAYER * LAYERS):.2f} GiB", flush=True)
    v0 = probe(0)
    v1 = probe(1)
    print(f"[mem48] 总判定 dev0={v0} dev1={v1}", flush=True)
