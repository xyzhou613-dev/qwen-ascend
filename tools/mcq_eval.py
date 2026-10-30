"""★ 任务级评测：24 道单项选择题，量"准确率"而不是 NLL。

为什么需要：`/fnx/loss` 是连续量、适合 A/B 同一配置，但**不能回答"用户觉得好不好"**。
这里用固定题库 + 强制只答字母，得到可直接对比的准确率（换配置/换量化/换层数都能跑）。

用法（宿主 python3，只走 HTTP）：
  python3 tools/mcq_eval.py                 # 默认关思考、每题 max_tokens=48
  python3 tools/mcq_eval.py --thinking      # 开思考对比
  python3 tools/mcq_eval.py --n 12          # 只跑前 12 题
  python3 tools/mcq_eval.py --verbose       # 打印每题回答
"""
import argparse
import json
import re
import sys
import time
import urllib.request

URL = "http://127.0.0.1:8000/v1/chat/completions"

# (题目, [(选项A, 选项B, ...)], 正确字母)
QUESTIONS = [
    ("中国的首都是？", ("上海", "北京", "广州", "深圳"), "B"),
    ("水的化学式是？", ("CO2", "H2O", "O2", "NaCl"), "B"),
    ("9 × 7 = ?", ("56", "63", "72", "49"), "B"),
    ("一年有多少个月？", ("10", "11", "12", "13"), "C"),
    ("太阳系中最大的行星是？", ("地球", "火星", "木星", "土星"), "C"),
    ("《论语》的作者（记录者）是孔子的？", ("朋友", "弟子", "政敌", "父亲"), "B"),
    ("二进制 1011 等于十进制？", ("9", "10", "11", "12"), "C"),
    ("下列哪个不是编程语言？", ("Python", "Java", "HTML 的标记", "C++"), "C"),
    ("人体最大的器官是？", ("肝脏", "皮肤", "大脑", "肺"), "B"),
    ("光在真空中的速度约为每秒？", ("3 万公里", "30 万公里", "300 万公里", "3000 公里"), "B"),
    ("下面哪个是质数？", ("21", "27", "29", "33"), "C"),
    ("中国哪个朝代最早？", ("秦", "汉", "商", "唐"), "C"),
    ("一个三角形内角和是多少度？", ("90", "180", "270", "360"), "B"),
    ("REST API 里用来获取资源的方法通常是？", ("POST", "GET", "DELETE", "PUT"), "B"),
    ("下列哪项属于监督学习？", ("K-means", "线性回归", "PCA", "DBSCAN"), "B"),
    ("1 加到 10 等于？", ("45", "50", "55", "66"), "C"),
    ("地球绕太阳一周约需？", ("24 小时", "30 天", "365 天", "10 年"), "C"),
    ("HTTP 状态码 404 表示？", ("服务器错误", "未找到", "无权限", "重定向"), "B"),
    ("下列哪个单位表示存储容量？", ("Hz", "GB", "m/s", "A"), "B"),
    ("“三人行，必有我师”出自？", ("《孟子》", "《论语》", "《庄子》", "《老子》"), "B"),
    ("神经网络中 ReLU 是？", ("损失函数", "激活函数", "优化器", "正则项"), "B"),
    ("地球上海洋面积占表面约？", ("30%", "50%", "71%", "90%"), "C"),
    ("下列哪项是数据库？", ("Nginx", "Redis", "Docker", "Git"), "B"),
    ("若 x + 5 = 12，则 x = ?", ("5", "6", "7", "8"), "C"),
]


def ask(q, opts, mx, thinking):
    letters = "ABCD"
    body_q = (f"{q}\n" + "\n".join(f"{letters[i]}. {o}" for i, o in enumerate(opts)) +
              "\n只回答正确选项的字母（单个字母），不要任何解释。")
    body = {"messages": [{"role": "user", "content": body_q}],
            "max_tokens": mx, "temperature": 0}
    if thinking:
        body["enable_thinking"] = True
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    d = json.load(urllib.request.urlopen(req, timeout=600))
    return d["choices"][0]["message"]["content"], time.perf_counter() - t


def pick(text):
    """从回答里抽第一个独立大写字母 A-D。"""
    for m in re.finditer(r"(?<![A-Za-z])([ABCD])(?![A-Za-z])", text.upper()):
        return m.group(1)
    return "?"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(QUESTIONS))
    ap.add_argument("--max_tokens", type=int, default=48)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    qs = QUESTIONS[:a.n]
    ok, t0 = 0, time.perf_counter()
    for i, (q, opts, key) in enumerate(qs, 1):
        txt, dt = ask(q, opts, a.max_tokens, a.thinking)
        got = pick(txt)
        good = got == key
        ok += good
        print(f"  {i:2d}. {'✓' if good else '✗'} 标的={key} 答={got}  {dt:5.2f}s  {q[:24]}",
              flush=True)
        if a.verbose and not good:
            print(f"        回答: {txt[:120]!r}", flush=True)
    n = len(qs)
    print(f"\n=== 准确率 {ok}/{n} = {ok/n*100:.1f}%   "
          f"平均 { (time.perf_counter()-t0)/n:.2f} s/题   "
          f"思考={'开' if a.thinking else '关'} ===")
    return 0 if ok == n else 0


if __name__ == "__main__":
    sys.exit(main())
