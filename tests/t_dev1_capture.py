"""★★★ 决定性隔离测试：**dev1 上到底能不能用 NPUGraph 捕获任何东西？**

之前 5 种组合都失败，但都混在"双卡 + 层 + 中转"的复杂实验里。这个脚本把它拆成梯子，
每级只有一处变量，每级都做"eager 基准 vs 图 replay"的数值对比：

  R1 平凡链，dev1，**current_device=1**（最纯的形态）
  R2 平凡链，dev1，current_device=0（显式指定设备张量）
  R3 平凡链，dev1，mode="global"（我们一直用 relaxed，换个模式）
  R4 平凡链，dev0（对照组，确认脚本本身没问题）
  R5 真层 L00，dev1（仅当 R1~R3 有任一通过时才跑）

判定：任一级 PASS 就说明"dev1 不是完全不能捕获"，问题在别处；
      R1~R3 全 FAIL 且 R4 PASS → **锁死：dev1 无法捕获**。
"""
import os
import sys

import torch

from fnx import engine as E

REL = "relaxed"
GLB = "global"
RES = []


def _trial(tag, dev, run, reset=None, cap_mode=REL, set_dev=None):
    """run() 往固定的 out 里写；对比 eager 与图 replay。"""
    old = torch.npu.current_device()
    try:
        if set_dev is not None:
            torch.npu.set_device(set_dev)
        sync = lambda: torch.npu.synchronize()
        s = torch.npu.Stream(device=dev) if set_dev is not None else torch.npu.Stream()
        # eager 基准
        if reset:
            reset()
        run()
        sync()
        ref = run.snapshot()
        # 预热 + 捕获
        with torch.npu.stream(s):
            for _ in range(3):
                run()
            sync()
            g = torch.npu.NPUGraph()
            try:
                with torch.npu.graph(g, capture_error_mode=cap_mode):
                    run()
            except Exception as e:  # noqa: BLE001
                print(f"  {tag:34s} 捕获失败 {type(e).__name__}: {str(e)[:70]}", flush=True)
                RES.append((tag, False))
                return
            sync()
        # replay
        if reset:
            reset()
        else:
            run.zero_() if hasattr(run, "zero_") else None
        run.clear()
        g.replay()
        sync()
        got = run.snapshot()
        d = (got.float() - ref.float()).abs()
        rel = float(d.mean() / ref.float().abs().mean().clamp_min(1e-12))
        ok = rel < 1e-3
        print(f"  {tag:34s} mode={cap_mode:8s} rel {rel*100:9.4f}%  "
              f"{'PASS ✅' if ok else 'FAIL ❌'}", flush=True)
        RES.append((tag, ok))
    except Exception as e:  # noqa: BLE001
        print(f"  {tag:34s} 异常 {type(e).__name__}: {str(e)[:70]}", flush=True)
        RES.append((tag, False))
    finally:
        try:
            torch.npu.set_device(old)
        except Exception:  # noqa: BLE001
            pass


class Chain:
    """最平凡的两算子链：out = x * 2 + 1"""

    def __init__(self, dev):
        self.x = torch.arange(1024, dtype=torch.float32, device=dev)
        self.out = torch.zeros(1024, dtype=torch.float32, device=dev)

    def __call__(self):
        self.out.copy_(self.x * 2.0 + 1.0)

    def clear(self):
        self.out.zero_()

    def snapshot(self):
        return self.out.clone()


def main():
    print(f"=== dev1 捕获隔离测试（current_device={torch.npu.current_device()}）===", flush=True)

    # R4 先做对照（dev0），确认脚本本身正确
    c0 = Chain("npu:0")
    _trial("R4 对照：平凡链 dev0", "npu:0", c0, cap_mode=REL, set_dev=0)

    # R1 dev1 + set_device(1)
    c1 = Chain("npu:1")
    _trial("R1 平凡链 dev1 (set_device=1)", "npu:1", c1, cap_mode=REL, set_dev=1)

    # R2 dev1，不切 current_device
    c2 = Chain("npu:1")
    _trial("R2 平凡链 dev1 (current=0)", "npu:1", c2, cap_mode=REL, set_dev=None)

    # R3 dev1 + global 模式
    c3 = Chain("npu:1")
    _trial("R3 平凡链 dev1 (mode=global)", "npu:1", c3, cap_mode=GLB, set_dev=1)

    ok = {t: v for t, v in RES}
    ctrl = ok.get("R4 对照：平凡链 dev0", False)
    dev1_any = any(v for t, v in RES if t.startswith("R1") or t.startswith("R2") or t.startswith("R3"))

    print("\n=== 判定 ===", flush=True)
    print(f"  对照组 dev0 : {'PASS' if ctrl else 'FAIL（脚本有问题，结论无效）'}", flush=True)
    print(f"  dev1 任一级 : {'PASS' if dev1_any else 'FAIL'}", flush=True)
    if ctrl and not dev1_any:
        print("  ★★ 锁死：**dev1 上 NPUGraph 无法捕获**（三种形态全失败，对照组通过）", flush=True)
        print("     → 910 上 dev1 只能 eager；现实路线改为「打包 + dev0 装 40 层」≈245 ms", flush=True)
    elif ctrl and dev1_any:
        print("  ★★ 好消息：dev1 **可以**捕获（某级通过）→ 继续上真层（R5）", flush=True)
    sys.exit(0 if (ctrl and dev1_any) else (0 if ctrl else 1))


if __name__ == "__main__":
    with torch.no_grad():
        main()
