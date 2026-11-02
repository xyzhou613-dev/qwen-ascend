"""子进程沙箱判题器, 对齐官方 lcb_runner 的判定语义.

- stdin   型 (atcoder/codeforces): 运行整份代码, 逐行比对 stdout
  (精确匹配失败时按 Decimal 逐元素比较, 兼容浮点/大整数).
- functional 型 (leetcode): exec 代码取出 Solution().<fn_name>,
  以 JSON 解析每行输入为调用参数, 比对返回值的 JSON 形式.

每个测试点在独立子进程中执行, 带超时 + 4GB 内存限制.
"""
import json
import os
import resource
import subprocess
import sys
import tempfile
from decimal import Decimal, InvalidOperation

RESULT_MARK = "__LCB_RESULT__"

# 与官方 testing_util.import_string 等价的导入前导
IMPORT_STRING = (
    "from string import *\nfrom re import *\nfrom datetime import *\n"
    "from collections import *\nfrom heapq import *\nfrom bisect import *\n"
    "from copy import *\nfrom math import *\nfrom random import *\n"
    "from statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\n"
    "from json import *\nfrom builtins import *\nfrom typing import *\n"
    "import string\nimport re\nimport datetime\nimport collections\n"
    "import heapq\nimport bisect\nimport copy\nimport math\nimport random\n"
    "import statistics\nimport itertools\nimport functools\nimport operator\n"
    "import io\nimport sys\nimport json\nsys.setrecursionlimit(50000)\n"
)

DRIVER_TAIL = """
import json as _json

_ns = {{}}
_src = {src!r}
exec(compile(_src, "<solution>", "exec"), _ns)

if "class Solution" in _src:
    _obj = _ns["Solution"]()
    _fn = getattr(_obj, {fn_name!r})
else:
    _fn = _ns[{fn_name!r}]

_args = [_json.loads(l) for l in {inp!r}.split("\\n")]
_res = _fn(*_args)
print("{mark}" + _json.dumps(_res, default=str))
"""


def _limits():
    resource.setrlimit(resource.RLIMIT_AS, (8 << 30, 8 << 30))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


# 直接 `python file.py` 时个别 Python 版本对含非 ASCII 的超长行
# 存在源文件解码 bug (实测 conda py3.9.13 误报 Non-UTF-8).
# 改用 bootstrap 在进程内显式以 utf-8 读码后 exec, 规避该路径.
_BOOTSTRAP = (
    "import sys\n"
    "exec(compile(open(sys.argv[1], encoding='utf-8').read(), "
    "sys.argv[1], 'exec'), {'__name__': '__main__'})\n"
)

_ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1")


def _run_py(code: str, stdin_data: str, timeout: int):
    """在子进程里执行 code, 返回 (stdout, err_type). err_type: None|'timeout'|'runtime'."""
    fd, path = tempfile.mkstemp(suffix=".py")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(code)
        p = subprocess.Popen(
            [sys.executable, "-c", _BOOTSTRAP, path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True,
            preexec_fn=_limits, env=_ENV,
        )
        try:
            out, _ = p.communicate(input=stdin_data, timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            return "", "timeout"
        if p.returncode != 0:
            return out, "runtime"
        return out, None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _stripped_lines(val):
    return [l.strip() for l in str(val).strip().split("\n")]


def _to_decimals(line):
    try:
        return True, [Decimal(x) for x in line.split()]
    except (InvalidOperation, ValueError):
        return False, []


def stdout_matches(prediction: str, expected: str) -> bool:
    """官方 grade_stdio 的输出比对逻辑."""
    pred_lines = _stripped_lines(prediction)
    gt_lines = _stripped_lines(expected)
    if len(pred_lines) != len(gt_lines):
        return False
    for pl, gl in zip(pred_lines, gt_lines):
        if pl == gl:
            continue
        ok1, dp = _to_decimals(pl)
        ok2, dg = _to_decimals(gl)
        if not (ok1 and ok2 and dp == dg):
            return False
    return True


def grade_stdin(code: str, tests: list, timeout: int):
    results = []
    for t in tests:
        out, err = _run_py(code, t["input"], timeout)
        if err == "timeout":
            results.append(("timeout", -3))
        elif err == "runtime":
            results.append(("runtime", -4))
        else:
            results.append(("ok" if stdout_matches(out, t["output"]) else "wa",
                            True if stdout_matches(out, t["output"]) else -2))
    return results


def grade_functional(code: str, tests: list, fn_name: str, timeout: int):
    results = []
    for t in tests:
        driver = DRIVER_TAIL.format(
            src=IMPORT_STRING + "\n" + code,
            fn_name=fn_name, inp=t["input"], mark=RESULT_MARK)
        out, err = _run_py(driver, "", timeout)
        if err:
            results.append((err, -3 if err == "timeout" else -4))
            continue
        pos = out.rfind(RESULT_MARK)
        if pos < 0:
            results.append(("runtime", -4))
            continue
        try:
            pred = json.loads(out[pos + len(RESULT_MARK):].strip())
            exp = json.loads(t["output"])
            results.append(("ok" if pred == exp else "wa",
                            True if pred == exp else -2))
        except Exception:
            results.append(("runtime", -4))
    return results


def grade_script(code: str, tests: list, test_imports: list, timeout: int):
    """EvalPlus 风格: 判题程序作为整体脚本运行 (HE+ 的 check(candidate) /
    MBPP+ 的顶层 assertion 循环), 退出码为 0 即通过."""
    imports = "\n".join(test_imports or [])
    results = []
    for t in tests:
        driver = (IMPORT_STRING + "\n" + imports + "\n" + code + "\n"
                  + t["input"] + "\n" + (t.get("call") or "") + "\n")
        _, err = _run_py(driver, "", timeout)
        if err:
            results.append((err, -3 if err == "timeout" else -4))
        else:
            results.append(("ok", True))
    return results


def grade_generation(problem: dict, code: str, timeout: int = 20):
    """对单条生成代码跑该题全部测试点.

    返回 (passed: bool, detail: list[(status, code)]).
    """
    tests = problem["tests"]
    fn_name = problem.get("func_name")
    if tests and tests[0].get("testtype") == "script":
        res = grade_script(code, tests, problem.get("test_imports"), timeout)
    elif fn_name:
        res = grade_functional(code, tests, fn_name, timeout)
    else:
        res = grade_stdin(code, tests, timeout)
    passed = all(r[1] is True for r in res)
    return passed, res
