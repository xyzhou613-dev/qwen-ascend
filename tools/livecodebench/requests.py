"""requests 的最小替身（纯标准库），供没有 requests / 没有外网的机器跑本评测。

只实现 run_eval.py 用到的 `requests.post(url, json=..., headers=..., timeout=...)`
以及返回值的 `.json()` / `.raise_for_status()` / `.status_code`。

放在 scripts/ 下即可，因为 `python3 scripts/run_eval.py` 的 sys.path[0] 就是 scripts/，
`import requests` 会先命中本文件（评测脚本本身无需改动）。
"""
import json as _json
import urllib.error
import urllib.request


class HTTPError(Exception):
    pass


class _Resp:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body

    def json(self):
        return _json.loads(self._body)

    def text(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise HTTPError(f"HTTP {self.status_code}: {self._body[:300]}")


def post(url, json=None, headers=None, timeout=None, **kwargs):  # noqa: A002
    data = _json.dumps(json).encode("utf-8") if json is not None else None
    h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return _Resp(r.status, r.read().decode("utf-8", "ignore"))
    except urllib.error.HTTPError as e:  # 4xx/5xx 也返回对象，交由 raise_for_status 处理
        return _Resp(e.code, e.read().decode("utf-8", "ignore"))
